"""Needle eval runner (see docs/eval_plan.md).

Runs every probe in eval/cases/ through the live backend's /debug/run pipeline,
judges the results (two-level: context-hit + answer-hit, plus hallucination on
negative probes), and writes a self-contained result JSON to eval/results/.

Usage:
  source .venv/bin/activate
  uvicorn main:app --reload          # backend must be running
  python eval/run.py --strategy concat
  python eval/run.py --strategy concat --case ml_001   # single case

The result JSON keeps every round's context_report so the debugger Eval tab
(and the future replay feature) can render without re-running anything.
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
import time

import httpx

# Use a plain chat model for evals, not supermind-agent-v1: the agent model runs
# its own internal loop upstream (usage can report 100k+ prompt tokens for a 2.5k
# context) and sometimes returns empty content, both of which pollute the metrics.
DEFAULT_MODEL = "deepseek-v4-pro"
# deepseek-v4-flash misjudges rubric compliance (verified false negatives); kimi-k2.5 is reliable
DEFAULT_JUDGE_MODEL = "kimi-k2.5"

# Bump when a case set changes; results from different versions are not comparable.
# smoke: 12 short cases (~1.5k tok histories) — fast regression / pipeline sanity.
# long:  6 cases (36-46 turns, ~8-10k tok, needle-carrying assistant replies
#        800+ chars with needles mid-reply, denser near-miss distractors) —
#        the tier that actually pressure-tests compression/chunking
#        strategies; see docs/eval_plan.md §6.6.
# tool-medium: medium history + frozen tool rounds (replayed, not re-executed);
#        isolates the loop-internal tool-result bloat dimension (§6.7).
# research: long history + multi-round tools — registered for §6.7 completeness;
#        cases not built yet (--tier research errors until eval/cases/research/ exists).
DATASET_VERSIONS = {
    "smoke": "v1-smoke",
    "long": "v1-long",
    "tool-medium": "v1-tool-medium",
    "research": "v1-research",
}

ANSWER_JUDGE_PROMPT = """你是评测裁判。下面是 AI 助手对用户问题的回答，请根据评分标准判断是否合格。

[用户问题]
{question}

[AI 回答]
{reply}

[评分标准]
{rubric}

只输出一个词：YES（合格）或 NO（不合格）。"""

FABRICATION_JUDGE_PROMPT = """你是评测裁判。AI 助手被问了一个它没有任何依据回答的问题。

[用户问题]
{question}

[AI 回答]
{reply}

该回答是否在没有依据的情况下编造了具体的事实性内容（具体数字、名称、日期等），\
而不是承认不知道、不确定或请求澄清？只输出一个词：YES（编造了）或 NO（没有编造）。"""

SUMMARY_PRESENCE_JUDGE_PROMPT = """下面是一段对话历史的压缩摘要。请判断这条信息是否在摘要中得到了保留。\
只要事实本身在摘要里出现，即便措辞、格式、标点不同（例如加了反引号、用"位于"代替括号、数字写法略有差异），也算保留。

[需要核对的信息]
{content}

[摘要文本]
{summary}

这条信息是否在摘要中得到保留？只输出一个词：YES（保留了）或 NO（没有保留）。"""


async def run_probe(client: httpx.AsyncClient, base_url: str, model: str,
                    strategy: str, history: list, question: str,
                    tool_rounds: list = None) -> dict:
    """Drive /debug/run via SSE; collect context_reports, usage, final reply.

    `tool_rounds` (tool tier): preset [assistant tool_call, tool result] pairs the
    backend replays frozen instead of executing tools (eval_plan §6.7 method A).
    """
    reports, usages, reply, error = [], [], None, None
    payload = {"model": model, "message": question, "history": history, "strategy": strategy}
    if tool_rounds:
        payload["tool_rounds"] = tool_rounds
    t0 = time.monotonic()
    async with client.stream(
        "POST", f"{base_url}/debug/run",
        json=payload,
        timeout=300.0,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "context_report":
                reports.append(event["data"])
            elif event["type"] == "round":
                usages.append(event["data"].get("usage") or {})
                if event["data"].get("finalReply") is not None:
                    reply = event["data"]["finalReply"]
            elif event["type"] == "final":
                usages.append(event.get("usage") or {})
                reply = event.get("reply") or reply
            elif event["type"] == "error":
                error = event.get("detail")
    latency = time.monotonic() - t0

    input_tokens = sum(u.get("prompt_tokens") or 0 for u in usages)
    output_tokens = sum(u.get("completion_tokens") or 0 for u in usages)
    usage_source = "api"
    if input_tokens == 0:  # upstream returned no usage — fall back to estimates
        input_tokens = sum(r["estimated_prompt_tokens"] for r in reports)
        usage_source = "estimated"

    return {
        "reply": reply, "error": error, "context_reports": reports,
        "rounds": len(reports), "latency_s": round(latency, 2),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "usage_source": usage_source,
    }


async def judge_yes(client: httpx.AsyncClient, base_url: str, judge_model: str,
                    prompt: str) -> tuple:
    """Returns (is_yes, raw_verdict) — raw kept in the result file for auditing."""
    resp = await client.post(
        f"{base_url}/debug/regenerate",
        json={"model": judge_model, "messages": [{"role": "user", "content": prompt}]},
        timeout=120.0,
    )
    resp.raise_for_status()
    raw = (resp.json().get("reply") or "").strip()
    first_line = raw.upper().splitlines()[0] if raw else ""
    return ("YES" in first_line and "NO" not in first_line.replace("YES", "")), raw


def _substring_present(content: str, text: str) -> bool:
    """Fast deterministic check: needle content literally present in text.
    Purely-numeric needles need a digit boundary ("242" must not match "2425")."""
    if re.fullmatch(r"\d+", content):
        return bool(re.search(r"(?<!\d)" + re.escape(content) + r"(?!\d)", text))
    return content in text


async def ctx_hit(needles: list, reports: list, *, client: httpx.AsyncClient,
                  base_url: str, judge_model: str) -> bool:
    """Were all the probe's needle messages effectively present in the context?

    Two judging modes (docs/phase1_sliding_window_summary.md §6):
      - Verbatim: the needle's original message position is in some layer's
        `msg_ids` (history/current/retrieved). This is concat's only mode and is
        unchanged — concat keeps every position, so its result is identical and
        never reaches an LLM call.
      - Summary-absorbed: the needle's position falls inside a `summary` layer's
        `covers_positions` range. Its raw position is gone, but the fact may
        survive in the summary TEXT. A rolling summary REFORMATS facts (backticks,
        rewording, punctuation), so an exact substring match false-negatives even
        when the fact is preserved. We therefore use substring only as a positive
        fast-path; when it fails we ask an LLM whether the fact is retained. This
        avoids the artifact where a correctly-summarized needle is scored as a miss.
    """
    included: set = set()
    summaries: list = []
    tool_texts: list = []
    for r in reports:
        for layer in r["layers"]:
            included.update(layer.get("msg_ids") or [])
            if layer.get("layer") == "summary":
                summaries.append((layer.get("covers_positions"), layer.get("text") or ""))
            elif layer.get("layer") == "tool_loop" and layer.get("text"):
                tool_texts.append(layer["text"])

    async def needle_present(n: dict) -> bool:
        # Tool-carrier needle (tool tier): no conversation `position` — it lives in a
        # frozen tool result. "In context" iff its content survived into a tool_loop
        # layer's text (deterministic substring; tool results aren't reformatted by
        # concat/window_summary, so no LLM fallback is needed here).
        if n.get("carrier") == "tool" or "position" not in n:
            content = str(n.get("content", "")).strip()
            return bool(content) and any(
                _substring_present(content, t) for t in tool_texts)
        pos = n["position"]
        if pos in included:
            return True
        content = str(n.get("content", "")).strip()
        if not content:
            return False
        covering = [text for cov, text in summaries if cov and cov[0] <= pos <= cov[1]]
        # Positive fast-path: a literal match is definitely present (no LLM needed).
        if any(_substring_present(content, text) for text in covering):
            return True
        # Substring failed — the summary may have reformatted the fact. Ask the LLM.
        for text in covering:
            is_yes, _ = await judge_yes(
                client, base_url, judge_model,
                SUMMARY_PRESENCE_JUDGE_PROMPT.format(content=content, summary=text))
            if is_yes:
                return True
        return False

    for n in needles:
        if not await needle_present(n):
            return False
    return True


def regex_judge(judge: dict, reply: str) -> bool:
    if not re.search(judge["pattern"], reply or ""):
        return False
    forbid = judge.get("forbid")
    if forbid and re.search(forbid, reply or ""):
        return False
    return True


def depth_bucket(position: int, conv_len: int) -> str:
    frac = position / max(conv_len, 1)
    return "early" if frac < 1 / 3 else "middle" if frac < 2 / 3 else "late"


def aggregate(probe_rows: list) -> dict:
    needle_rows = [p for p in probe_rows if p["style"] != "negative"]

    def rate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def avg(rows, key, digits=1):
        return round(sum(r[key] for r in rows) / len(rows), digits) if rows else None

    summary = {
        "probes": len(probe_rows),
        "needle_probes": len(needle_rows),
        "negative_probes": len(probe_rows) - len(needle_rows),
        "ctx_hit_rate": rate(needle_rows, "ctx_hit"),
        "ans_hit_rate": rate(needle_rows, "ans_hit"),
        # hallucination_rate uses all probes as denominator so it stays
        # comparable across strategies with different ctx-miss counts
        "hallucination_rate": round(
            sum(1 for p in probe_rows if p["hallucination"]) / len(probe_rows), 4),
        "avg_input_tokens": avg(probe_rows, "input_tokens"),
        # estimated twin is robust to upstream usage anomalies (api occasionally
        # reports 100k+ for a 2k context when the provider runs internal tooling)
        "avg_estimated_input_tokens": avg(probe_rows, "estimated_input_tokens"),
        "avg_output_tokens": avg(probe_rows, "output_tokens"),
        "avg_rounds": avg(probe_rows, "rounds"),
        "avg_latency_s": avg(probe_rows, "latency_s"),
        "avg_compression_ratio": rate(probe_rows, "compression_ratio"),
    }

    dimensions: dict = {}
    for dim in ("style", "carrier", "depth", "needle_type", "topic"):
        buckets: dict = {}
        for p in probe_rows:
            b = p.get(dim)
            if b is None:
                continue
            buckets.setdefault(b, []).append(p)
        dimensions[dim] = {
            b: {
                "probes": len(rows),
                "ctx_hit_rate": rate(rows, "ctx_hit"),
                "ans_hit_rate": rate(rows, "ans_hit"),
                "hallucination_rate": round(
                    sum(1 for r in rows if r["hallucination"]) / len(rows), 4),
            }
            for b, rows in buckets.items()
        }
    return {"summary": summary, "dimensions": dimensions}


async def main() -> None:
    ap = argparse.ArgumentParser(description="Run the needle eval set.")
    ap.add_argument("--strategy", default="concat")
    ap.add_argument("--tier", default="smoke", choices=sorted(DATASET_VERSIONS),
                    help="which case set to run (cases/<tier>/)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--cases", help="override the cases directory (default: cases/<tier>/)")
    ap.add_argument("--case", help="run a single case id, e.g. ml_001")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    if not args.cases:
        args.cases = os.path.join(os.path.dirname(__file__), "cases", args.tier)

    case_files = sorted(
        f for f in os.listdir(args.cases)
        if f.endswith(".json") and (not args.case or f == f"{args.case}.json")
    )
    if not case_files:
        sys.exit(f"No case files found in {args.cases}")

    probe_rows = []
    async with httpx.AsyncClient() as client:
        for fname in case_files:
            with open(os.path.join(args.cases, fname), encoding="utf-8") as f:
                case = json.load(f)
            conv = case["conversation"]
            case_needles = {n["id"]: n for n in case.get("needles", [])}

            for probe in case["probes"]:
                label = f"{case['id']}/{probe['id']}"
                print(f"[{label}] {probe['question'][:40]}…", flush=True)
                run = await run_probe(client, args.base_url, args.model,
                                      args.strategy, conv, probe["question"],
                                      tool_rounds=probe.get("tool_rounds"))
                if run["error"]:
                    print(f"  ERROR: {run['error']}", flush=True)

                # Tool-tier probes carry their needles inline (they live in this
                # probe's frozen tool rounds); resolve ids against the union.
                needles = {**case_needles, **{n["id"]: n for n in probe.get("needles", [])}}
                probe_needles = [needles[nid] for nid in probe["needle_ids"]]
                is_negative = probe["style"] == "negative"
                hit_ctx = None if is_negative else await ctx_hit(
                    probe_needles, run["context_reports"],
                    client=client, base_url=args.base_url, judge_model=args.judge_model)

                judge = probe["judge"]
                judge_detail = judge["method"]
                judge_raw = None
                if judge["method"] == "regex":
                    hit_ans = regex_judge(judge, run["reply"])
                else:
                    hit_ans, judge_raw = await judge_yes(
                        client, args.base_url, args.judge_model,
                        ANSWER_JUDGE_PROMPT.format(
                            question=probe["question"], reply=run["reply"] or "",
                            rubric=judge["rubric"]))

                # Hallucination: negative probe answered with fabricated specifics,
                # or a needle probe that MISSED context AND gave a wrong answer the
                # model nonetheless stated as confident fact. A correct answer is
                # never a hallucination, so `not hit_ans` guards the needle branch —
                # without it, a needle correctly recalled from the summary (ctx-miss
                # by the strict position check) would be mislabeled as fabrication.
                hallucination = False
                if is_negative:
                    hallucination = not hit_ans
                elif hit_ctx is False and not hit_ans:
                    hallucination, _ = await judge_yes(
                        client, args.base_url, args.judge_model,
                        FABRICATION_JUDGE_PROMPT.format(
                            question=probe["question"], reply=run["reply"] or ""))

                first_report = run["context_reports"][0] if run["context_reports"] else None
                compression = None
                if first_report and first_report["full_history_tokens"]:
                    # History-compression ratio: the history-derived layers actually
                    # sent (summary + verbatim window) over the full history. `current`
                    # is excluded — it's the new turn, identical across strategies and
                    # not part of the history denominator — so concat stays exactly 1.0
                    # and window_summary reads as the fraction of history retained.
                    sent = sum(
                        l["tokens"] for l in first_report["layers"]
                        if l["layer"] in ("summary", "history"))
                    compression = round(sent / first_report["full_history_tokens"], 4)

                first_needle = probe_needles[0] if probe_needles else None
                probe_rows.append({
                    "case_id": case["id"], "probe_id": probe["id"],
                    "topic": case["topic"], "style": probe["style"],
                    "carrier": first_needle.get("carrier") if first_needle else None,
                    "needle_type": first_needle.get("type") if first_needle else None,
                    # tool-carrier needles have no conversation depth (`tool_round`,
                    # not `position`) — leave depth unbucketed for them.
                    "depth": depth_bucket(first_needle["position"], len(conv))
                    if first_needle and "position" in first_needle else None,
                    "question": probe["question"],
                    "reply": run["reply"], "error": run["error"],
                    "ctx_hit": hit_ctx, "ans_hit": hit_ans, "hallucination": hallucination,
                    "judge_detail": judge_detail, "judge_raw": judge_raw,
                    "input_tokens": run["input_tokens"], "output_tokens": run["output_tokens"],
                    "estimated_input_tokens": sum(
                        r["estimated_prompt_tokens"] for r in run["context_reports"]),
                    "usage_source": run["usage_source"],
                    "rounds": run["rounds"], "latency_s": run["latency_s"],
                    "compression_ratio": compression,
                    "context_reports": run["context_reports"],
                })
                mark = lambda v: "n/a" if v is None else ("✓" if v else "✗")
                print(f"  ctx {mark(hit_ctx)} · ans {mark(hit_ans)}"
                      f"{' · HALLUCINATION' if hallucination else ''}"
                      f" · in {run['input_tokens']} tok · {run['latency_s']}s", flush=True)

    now = datetime.datetime.now()
    run_id = f"{now:%Y-%m-%d_%H%M}_{args.strategy}_{args.tier}"
    result = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "model": args.model,
        "judge_model": args.judge_model,
        "dataset_version": DATASET_VERSIONS[args.tier],
        **aggregate(probe_rows),
        "probes": probe_rows,
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    s = result["summary"]
    print(f"\n{'='*60}\n{run_id}")
    print(f"ctx hit {s['ctx_hit_rate']} · ans hit {s['ans_hit_rate']}"
          f" · halluc {s['hallucination_rate']}")
    print(f"avg input {s['avg_input_tokens']} tok · output {s['avg_output_tokens']} tok"
          f" · rounds {s['avg_rounds']} · latency {s['avg_latency_s']}s")
    print(f"→ {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
