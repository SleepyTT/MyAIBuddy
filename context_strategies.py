"""Context assembly strategies (see docs/phase1_sliding_window_summary.md).

A single `assemble_context(...)` is shared by BOTH `/chat` (production, reads
history from the DB by chat_id) and `/debug/run` (stateless, receives history
directly from the eval runner) so that what eval measures and what production
sends are assembled identically.

Strategies
----------
- "concat":         brute-force baseline. Send the whole history verbatim,
                    then the current user message. This is the accuracy ceiling.
- "window_summary": keep the last `WINDOW_TURNS` history messages verbatim and
                    compress everything older into a single rolling summary
                    placed right after the system prompt (here: at the front,
                    we have no system prompt in the upstream contract). When the
                    history is short enough to fit the window, the summary is
                    empty and the behaviour is identical to concat.

The summary is produced by one cheap upstream LLM call. For `/chat` the summary
is persisted on the Chat row and updated incrementally (old summary + the turns
that just fell out of the window -> new summary). For `/debug/run` there is no
DB, so the summary is computed on the fly from the passed history.
"""

import os
from typing import Any, Callable, List, Optional, Tuple

import httpx

AI_BUILDER_BASE_URL = "https://space.ai-builders.com/backend/v1"

# Phase 1 knobs (docs/phase1 §2). Tunable via env for the N∈{4,6,10} sweep.
WINDOW_TURNS = int(os.getenv("WINDOW_TURNS", "6"))          # last N history msgs kept verbatim
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "400"))  # rolling summary budget
# A cheap, fast model for the summary call (cost is "management overhead").
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek-v4-flash")

ALLOWED_STRATEGIES = {"concat", "window_summary"}

SUMMARY_SYSTEM_PROMPT = (
    "将以下对话压缩成不超过 400 字的要点摘要，保留所有具体数值、决定、专有名词、"
    "函数名、端口号、日期等精确信息。如果某事实后来被更正，只保留最新值并标明它是最新结论。"
    "只输出摘要正文，不要任何前言或解释。"
)


def split_window(history: List[dict], window_turns: int = WINDOW_TURNS
                 ) -> Tuple[List[dict], List[dict], int]:
    """Split history into (older, window, split_index).

    `older` are the messages that fall *outside* the verbatim window and must be
    absorbed into the summary. `window` are the last `window_turns` messages kept
    verbatim. `split_index` is the position (index into `history`) of the first
    in-window message — i.e. older = history[:split_index].

    When history fits inside the window, `older` is empty and `split_index ==
    len(history)` ... no, it's the start of the window which is 0; we return
    split_index = max(0, len(history) - window_turns).
    """
    n = len(history)
    split_index = max(0, n - window_turns)
    return history[:split_index], history[split_index:], split_index


def _render_turns_for_summary(turns: List[dict]) -> str:
    lines = []
    for m in turns:
        role = m.get("role", "user")
        content = m.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def summarize_increment(
    old_summary: Optional[str],
    new_turns: List[dict],
    *,
    api_key: str,
    model: str = SUMMARY_MODEL,
    base_url: str = AI_BUILDER_BASE_URL,
) -> Tuple[str, int]:
    """Incrementally fold `new_turns` into `old_summary` via one upstream call.

    Returns (new_summary_text, summary_cost_tokens). `summary_cost_tokens` is the
    upstream prompt+completion usage for this management call (0 if usage absent).
    If there are no new turns to absorb, returns the old summary unchanged with
    zero cost (no call made).
    """
    if not new_turns:
        return (old_summary or ""), 0

    prefix = ""
    if old_summary:
        prefix = f"已有摘要（请在此基础上合并新内容）：\n{old_summary}\n\n新增对话：\n"
    else:
        prefix = "对话：\n"
    user_content = prefix + _render_turns_for_summary(new_turns)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError:
        # Upstream network/5xx failure: degrade to window-only context rather
        # than aborting the whole turn. Keep the old summary, charge no cost.
        return (old_summary or ""), 0
    new_summary = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage") or {}
    cost = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    # Fall back to keeping the old summary if the call returned empty.
    if not new_summary:
        new_summary = old_summary or ""
    return new_summary, cost


async def ensure_summary(
    history: List[dict],
    *,
    api_key: str,
    summary: Optional[str] = None,
    summary_through_position: int = 0,
    window_turns: int = WINDOW_TURNS,
    model: str = SUMMARY_MODEL,
    base_url: str = AI_BUILDER_BASE_URL,
) -> Tuple[str, int, int]:
    """Bring the rolling summary up to date with the current window boundary.

    Given the persisted `summary` covering history positions [0, summary_through_position),
    fold in any turns that have since dropped out of the verbatim window. Returns
    (summary_text, new_summary_through_position, summary_cost_tokens).

    `new_summary_through_position` is the index of the first in-window message
    (== `split_index` from split_window). When history fits the window it is 0
    and the summary is left empty.
    """
    _, _, split_index = split_window(history, window_turns)
    if split_index <= summary_through_position:
        # Summary already covers everything outside the window (or window covers all).
        return (summary or ""), summary_through_position, 0
    newly_dropped = history[summary_through_position:split_index]
    new_summary, cost = await summarize_increment(
        summary, newly_dropped, api_key=api_key, model=model, base_url=base_url
    )
    return new_summary, split_index, cost


def assemble_context(
    strategy: str,
    history: List[dict],
    current_msg: dict,
    *,
    summary: Optional[str] = None,
    summary_through_position: Optional[int] = None,
    window_turns: int = WINDOW_TURNS,
) -> dict:
    """Assemble the message list to send upstream for one strategy.

    Returns a dict describing the assembled context so callers (both /chat and
    /debug/run) and the context_report builder stay in sync:

        {
          "messages": [...],              # full message list to send upstream
          "strategy": str,
          "summary": str | None,          # the summary text used (window_summary)
          "summary_through_position": int,# first in-window history index
          "window_ids": [int, ...],       # history indices kept verbatim in-window
          "included_history_ids": [int],  # history indices physically present
          "dropped_turns": int,           # history turns absorbed by the summary
          "window_turns": int,
        }

    NOTE: this function does NOT make the summary LLM call. For window_summary,
    pass a precomputed `summary` (+ `summary_through_position`); callers use
    `ensure_summary(...)` (async, makes the cheap call) beforehand. When no
    summary is supplied, the older turns are simply dropped (callers are expected
    to have summarized first — this fallback keeps assembly pure/synchronous).
    """
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unknown context strategy: {strategy}")

    if strategy == "concat":
        messages = [*history, current_msg]
        ids = list(range(len(history)))
        return {
            "messages": messages,
            "strategy": strategy,
            "summary": None,
            "summary_through_position": 0,
            "window_ids": ids,
            "included_history_ids": ids,
            "dropped_turns": 0,
            "window_turns": window_turns,
        }

    # strategy == "window_summary"
    older, window, split_index = split_window(history, window_turns)
    # If a summary_through_position was supplied, trust it; else it equals the
    # window boundary (everything older is summarized).
    through = summary_through_position if summary_through_position is not None else split_index

    window_ids = list(range(split_index, len(history)))
    messages: List[dict[str, Any]] = []
    summary_text = (summary or "").strip()
    if summary_text:
        messages.append({
            "role": "system",
            "content": f"以下是更早对话的摘要，供参考：\n{summary_text}",
        })
    messages.extend(window)
    messages.append(current_msg)

    return {
        "messages": messages,
        "strategy": strategy,
        "summary": summary_text or None,
        "summary_through_position": through,
        "window_ids": window_ids,
        "included_history_ids": window_ids,  # only in-window turns are verbatim
        "dropped_turns": split_index,        # turns absorbed by the summary
        "window_turns": window_turns,
    }
