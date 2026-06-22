"""Structural validator for the tool-medium eval cases (docs/eval_plan.md §6.7/§6.8).

Pure/offline (no network): checks the frozen JSON against the tier's contract so a
regeneration can be sanity-checked in one command. Run:

    source .venv/bin/activate
    python eval/cases/tool-medium/_validate.py

Exits non-zero if any case fails. Companion to `_generate.py` (which captures the
REAL read_page/web_search output via the app tools — see that file's docstring).
"""
import json
import os
import re
import sys

import tiktoken

DIR = os.path.dirname(os.path.abspath(__file__))
ENC = tiktoken.get_encoding("cl100k_base")
CONV_TOK_MIN = 2000          # clearly above smoke (~1.5k); medium, not long (~8.7k)
READ_PAGE_MIN, READ_PAGE_MAX = 4000, 5000
NEEDLE_FRAC_LO, NEEDLE_FRAC_HI = 0.12, 0.88  # needle must sit mid-result


def toks(s):
    return len(ENC.encode(s))


def substr(content, text):
    """Mirror eval/run.py `_substring_present`: digit-boundary for pure-numeric."""
    if re.fullmatch(r"\d+", content):
        return bool(re.search(r"(?<!\d)" + re.escape(content) + r"(?!\d)", text))
    return content in text


def main():
    cov = {"topics": set(), "styles": set(), "carriers": set(), "types": set(),
           "multi_round": 0, "negative": 0, "two_needle": 0, "tool_probes": 0}
    allok = True
    print(f"{'case':18}{'conv_tok':>9}{'msgs':>5}  checks")
    for fn in sorted(f for f in os.listdir(DIR) if f.endswith(".json")):
        c = json.load(open(os.path.join(DIR, fn), encoding="utf-8"))
        cov["topics"].add(c["topic"])
        txt = "\n".join(m["content"] or "" for m in c["conversation"])
        ct = toks(txt)
        tail = txt[int(len(txt) * 0.75):]
        iss = []
        if ct < CONV_TOK_MIN:
            iss.append(f"CONV_TOK={ct}<{CONV_TOK_MIN}")
        for n in c.get("needles", []):  # case-level (conversation) needles
            if n["content"] not in c["conversation"][n["position"]]["content"]:
                iss.append(f"caseN {n['id']} !@pos{n['position']}")
            cov["carriers"].add(n["carrier"]); cov["types"].add(n["type"])
        for p in c["probes"]:
            cov["styles"].add(p["style"])
            if p["style"] == "negative":
                cov["negative"] += 1
            if "needle_ids" not in p:
                iss.append(f"{p['id']} missing needle_ids")
            tr = p.get("tool_rounds", [])
            pn = p.get("needles", [])
            if tr:
                cov["tool_probes"] += 1
            if len(tr) >= 2:
                cov["multi_round"] += 1
            if sum(1 for x in pn if x.get("carrier") == "tool") >= 2:
                cov["two_needle"] += 1
            for i, r in enumerate(tr):
                tc = r["assistant"]["tool_calls"][0]
                if tc["id"] != r["tool"]["tool_call_id"]:
                    iss.append(f"{p['id']}r{i} tool_call_id mismatch")
                try:
                    json.loads(tc["function"]["arguments"])
                except Exception:
                    iss.append(f"{p['id']}r{i} arguments not JSON")
                if tc["function"]["name"] == "read_page" and not (
                        READ_PAGE_MIN <= len(r["tool"]["content"]) <= READ_PAGE_MAX):
                    iss.append(f"{p['id']}r{i} read_page len={len(r['tool']['content'])}")
            for nd in pn:
                cov["carriers"].add(nd["carrier"]); cov["types"].add(nd["type"])
                if nd.get("carrier") == "tool":
                    tcont = tr[nd["tool_round"]]["tool"]["content"]
                    idx = tcont.find(nd["content"])
                    if idx < 0:
                        iss.append(f"{p['id']} {nd['id']} ABSENT in tool_round {nd['tool_round']}")
                    elif not (NEEDLE_FRAC_LO <= idx / len(tcont) <= NEEDLE_FRAC_HI):
                        iss.append(f"{p['id']} {nd['id']} not-mid (frac={idx/len(tcont):.2f})")
                    if substr(nd["content"], tail):
                        iss.append(f"{p['id']} {nd['id']} LEAK in conv tail")
            if p["judge"]["method"] == "regex" and pn:
                tcont = tr[pn[0]["tool_round"]]["tool"]["content"]
                if not re.search(p["judge"]["pattern"], tcont):
                    iss.append(f"{p['id']} regex doesn't match its tool result")
        if iss:
            allok = False
        print(f"{c['id']:18}{ct:>9}{len(c['conversation']):>5}  {'OK' if not iss else '; '.join(iss)}")
    print("\nCOVERAGE:", {k: (sorted(v) if isinstance(v, set) else v) for k, v in cov.items()})
    # Coverage gates (eval_plan §6.6/§6.7)
    gate = []
    if len(cov["topics"]) < 5: gate.append("missing topics")
    if not {"direct", "paraphrase", "implicit", "negative"} <= cov["styles"]: gate.append("missing a style")
    if cov["negative"] < 2: gate.append("<2 negatives")
    if cov["multi_round"] < 1: gate.append("no multi-round probe")
    if cov["two_needle"] < 1: gate.append("no two-needle multi-hop probe")
    if gate:
        allok = False
        print("COVERAGE GAPS:", gate)
    print("\n*** ALL PASS ***" if allok else "\n*** ISSUES ***")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
