"""Token estimation and context_report generation (see docs/eval_plan.md §3).

Dual-track token counting: local tiktoken (cl100k_base) estimates give the
per-layer breakdown; the upstream API `usage` field is the ground-truth total.
The upstream tokenizer is unknown, so estimates are for relative comparison.
"""

import json
from functools import lru_cache
from typing import Any, List, Optional

CONTEXT_LIMIT = 128_000  # assumed upstream context window

# Rough per-message wrapper cost (role markers etc.) in OpenAI-style chat format
PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: Any) -> int:
    if not text:
        return 0
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    try:
        return len(_encoder().encode(s))
    except Exception:
        # tiktoken unavailable (e.g. encoding file not downloadable): chars/3
        # is a usable approximation for mixed Chinese/English text.
        return max(1, len(s) // 3)


def message_tokens(msg: dict) -> int:
    total = PER_MESSAGE_OVERHEAD
    total += estimate_tokens(msg.get("content"))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        total += estimate_tokens(fn.get("name")) + estimate_tokens(fn.get("arguments"))
    return total


def _layer(name: str, msgs: List[dict], msg_ids: Optional[List[int]] = None) -> dict:
    layer: dict[str, Any] = {
        "layer": name,
        "tokens": sum(message_tokens(m) for m in msgs),
        "items": len(msgs),
    }
    if msg_ids is not None:
        layer["msg_ids"] = msg_ids
    return layer


def build_context_report(
    round_num: int,
    strategy: str,
    history: List[dict],
    included_history_ids: List[int],
    current: dict,
    loop_messages: List[dict],
    tools: List[dict],
    summary: Optional[str] = None,
    summary_through_position: int = 0,
    summary_cost_tokens: int = 0,
    context_limit: int = CONTEXT_LIMIT,
) -> dict:
    """Describe exactly what is being sent to the LLM this round.

    `history` is the full prior conversation (stable msg id = list index);
    `included_history_ids` are the ids the strategy kept verbatim in the context
    — for the brute-force concat baseline that is all of them; for window_summary
    it is only the in-window turns. `loop_messages` are the assistant tool-calls
    / tool results accumulated during this question's agentic loop (empty in
    round 1).

    Phase 1 (window_summary): when `summary` is non-empty a `summary` layer is
    added (with `covers_positions = [0, summary_through_position - 1]`), the
    `history` layer counts only the in-window turns, and `dropped_turns` /
    `summary_cost_tokens` report the management cost (docs/phase1 §5).
    """
    layers = []

    summary_text = (summary or "").strip()
    if summary_text:
        layers.append({
            "layer": "summary",
            "tokens": estimate_tokens(summary_text) + PER_MESSAGE_OVERHEAD,
            "items": 1,
            "covers_positions": [0, max(0, summary_through_position - 1)],
            # The summary text is carried so the eval ctx_hit judge can check
            # whether an absorbed needle's content survived compression, and so
            # the debugger can render it on replay.
            "text": summary_text,
        })

    included = [history[i] for i in included_history_ids]
    layers.append(_layer("history", included, included_history_ids))
    layers.append(_layer("current", [current]))

    if loop_messages:
        tool_loop_layer = _layer("tool_loop", loop_messages)
        # Carry the tool-result text so the eval ctx_hit judge can check whether a
        # needle embedded in a (possibly bloated) tool result survived into this
        # round's context. Mirrors how the `summary` layer carries `text`.
        tool_texts = [
            m.get("content") for m in loop_messages
            if m.get("role") == "tool" and m.get("content")
        ]
        if tool_texts:
            tool_loop_layer["text"] = "\n".join(
                t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
                for t in tool_texts
            )
        layers.append(tool_loop_layer)
    tools_tokens = estimate_tokens(tools) if tools else 0
    layers.append({"layer": "tools_schema", "tokens": tools_tokens, "items": len(tools)})

    full_history_tokens = sum(message_tokens(m) for m in history)
    estimated = sum(l["tokens"] for l in layers)
    # dropped_turns: history turns absorbed by the summary (not in the window).
    dropped_turns = summary_through_position if summary_text else 0
    return {
        "round": round_num,
        "strategy": strategy,
        "layers": layers,
        "estimated_prompt_tokens": estimated,
        "full_history_tokens": full_history_tokens,
        "context_limit": context_limit,
        "dropped_turns": dropped_turns,
        "summary_cost_tokens": summary_cost_tokens,
        "candidates": [],  # Phase 2 (RAG): retrieval candidates incl. rejected
    }
