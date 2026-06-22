"""End-to-end smoke test for the tool-medium frozen-replay path.

Needs the backend running (uvicorn main:app). For every tool probe it drives
/debug/run with the frozen tool_rounds and checks the replay mechanics + that the
needle survives into context and (for the concat ceiling) is surfaced in the answer.

    source .venv/bin/activate
    uvicorn main:app --port 8000        # in another shell
    python eval/cases/tool-medium/_smoketest.py --base-url http://localhost:8000

This drives the REAL upstream LLM, so it costs upstream calls and the LLM-judged
answer checks are advisory (the deterministic ctx/replay checks are the gate).
"""
import argparse
import json
import os
import re
import sys

import httpx

DIR = os.path.dirname(os.path.abspath(__file__))


def answer_token(needle_content):
    """The salient answer value buried in a (long, English) ctx-match needle.

    Needle `content` is a unique phrase chosen for deterministic ctx_hit (e.g.
    "revenue came in at 475"); the value a Chinese reply actually echoes is the
    number/percent/version/identifier inside it. Extract that for the advisory
    answer check (the authoritative answer judging lives in eval/run.py)."""
    # value tokens first (version, percent, snake_case identifier, number), so a
    # leading prose word ("revenue came in at 475") never shadows the value (475).
    m = re.search(r"\d+\.\d+\.\d+|\d+\s?%|[a-z]+(?:_[a-z]+)+|\d{2,}|\b\d\b", needle_content)
    return m.group(0) if m else needle_content


def drive(base_url, model, history, question, tool_rounds):
    reports, final, err = [], None, None
    body = {"model": model, "message": question, "history": history,
            "strategy": "concat", "tool_rounds": tool_rounds}
    with httpx.Client(timeout=180) as c:
        with c.stream("POST", f"{base_url}/debug/run", json=body) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ev["type"] == "context_report":
                        reports.append(ev["data"])
                    elif ev["type"] == "final":
                        final = ev
                    elif ev["type"] == "error":
                        err = ev.get("detail")
    return reports, final, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    failures = 0
    for fn in sorted(f for f in os.listdir(DIR) if f.endswith(".json")):
        case = json.load(open(os.path.join(DIR, fn), encoding="utf-8"))
        for p in case["probes"]:
            tr = p.get("tool_rounds")
            if not tr:
                continue
            reports, final, err = drive(args.base_url, args.model,
                                        case["conversation"], p["question"], tr)
            tag = f"{case['id']}/{p['id']}"
            if err or final is None:
                print(f"[FAIL] {tag}: error={err}")
                failures += 1
                continue
            # Deterministic gate: exactly one context_report, tool_loop carries needles.
            tl = next((l for r in reports for l in r["layers"]
                       if l["layer"] == "tool_loop"), None)
            reply = final.get("reply") or ""
            ctx_ok = len(reports) == 1 and tl is not None
            needle_in_ctx = tl and all(nd["content"] in (tl.get("text") or "")
                                       for nd in p.get("needles", []))
            est = reports[0]["estimated_prompt_tokens"] if reports else 0
            # Advisory: did the (concat-ceiling) answer surface each needle's value?
            ans = [nd for nd in p.get("needles", [])
                   if answer_token(nd["content"]) in reply]
            status = "OK" if (ctx_ok and needle_in_ctx) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(f"[{status}] {tag}: reports={len(reports)} est_input={est} tok "
                  f"needle_in_ctx={bool(needle_in_ctx)} ans_surfaced={len(ans)}/{len(p.get('needles', []))}")
    print("\nALL REPLAY CHECKS PASS" if failures == 0 else f"\n{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
