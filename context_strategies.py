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

from context_report import estimate_tokens, message_tokens  # token accounting (no cycle)

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


# ── Phase 1.5: budget-aware assembly (docs/phase1.5_context_budget.md) ───────
# Make assembly respect a token budget B. Priority: current question + tools
# schema (always) > loop tool results (high) > recent history (window) > older
# history (summary for window_summary / dropped for concat). At a budget large
# enough to hold everything, output is identical to the non-budgeted path above.

CONTEXT_BUDGET_DEFAULT = 128_000  # default token budget == full window → no-op


def _truncate_to_tokens(content: str, token_budget: int) -> Tuple[str, bool]:
    """Head-truncate a string so a message holding it fits `token_budget` tokens.

    Returns (content, truncated). Head-truncation (keep the start) mirrors
    PAGE_TEXT_LIMIT; our tool needles sit mid-result, so a tight budget cuts them.
    """
    if estimate_tokens(content) + 4 <= token_budget:
        return content, False
    if token_budget <= 4:
        return "", True
    target = token_budget - 4
    lo, hi = 0, len(content)
    while hi - lo > 40:
        mid = (lo + hi) // 2
        if estimate_tokens(content[:mid]) <= target:
            lo = mid
        else:
            hi = mid
    return content[:lo], True


def _concat_window_capped(history: List[dict], token_budget: int) -> Tuple[List[dict], List[dict], int]:
    """Newest suffix of history fitting `token_budget` (concat under budget: drop older)."""
    start = len(history)
    total = 0
    while start > 0:
        t = message_tokens(history[start - 1])
        if total + t > token_budget:
            break
        total += t
        start -= 1
    return history[:start], history[start:], start  # (dropped, kept, split_index)


def _window_capped(history: List[dict], window_turns: int, token_budget: int
                   ) -> Tuple[List[dict], List[dict], int]:
    """Last `window_turns` messages, further trimmed oldest-first to fit `token_budget`."""
    start = max(0, len(history) - window_turns)  # WINDOW_TURNS cap → default behavior at large budget
    window = history[start:]
    while window and sum(message_tokens(m) for m in window) > token_budget:
        window = window[1:]
        start += 1
    return history[:start], window, start


async def summarize_text(text: str, *, api_key: str, model: str = SUMMARY_MODEL,
                         base_url: str = AI_BUILDER_BASE_URL) -> Tuple[str, int]:
    """Query-blind compression of one big blob (an over-budget tool result),
    preserving specifics. Returns (summary, cost_tokens); ("", 0) on failure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content":
                "将以下网页/工具返回内容压缩成要点，保留所有具体数值、名称、版本号、函数名、"
                "端口、日期等精确信息，去掉导航和无关样板。只输出要点正文。"},
            {"role": "user", "content": text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError:
        return "", 0
    out = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage") or {}
    return out, (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)


async def assemble_budgeted(
    strategy: str,
    history: List[dict],
    current_msg: dict,
    loop_messages: List[dict],
    tools: List[dict],
    budget: int,
    *,
    api_key: str,
    model: str = SUMMARY_MODEL,
    base_url: str = AI_BUILDER_BASE_URL,
    window_turns: int = WINDOW_TURNS,
) -> dict:
    """Assemble one upstream prompt under token `budget` (Phase 1.5).

    Returns the same shape /debug/run needs, plus `base_len` (messages[:base_len]
    is summary+window+current; messages[base_len:] are the fitted tool results).
    """
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unknown context strategy: {strategy}")

    # current question + tools schema are never compressed (priority "必保"), so the
    # budget is a SOFT floor: below `reserved` the prompt can exceed `budget`. The
    # documented sweep floor (4k) is well above `reserved`, so this never bites.
    reserved = message_tokens(current_msg) + (estimate_tokens(tools) if tools else 0)
    avail = max(0, budget - reserved)

    # 1) Loop tool results first (they're why the model is answering this turn).
    fitted_loop: List[dict] = []
    summary_cost = 0
    loop_used = 0
    for m in loop_messages:
        if m.get("role") != "tool":
            fitted_loop.append(m)
            loop_used += message_tokens(m)
            continue
        remaining = max(0, avail - loop_used)
        if message_tokens(m) <= remaining:
            fitted_loop.append(m)
            loop_used += message_tokens(m)
            continue
        content = m.get("content") or ""
        if strategy == "window_summary":  # summarize the over-budget tool result
            summ, cost = await summarize_text(content, api_key=api_key, model=model, base_url=base_url)
            summary_cost += cost
            content, _ = _truncate_to_tokens(summ or content, remaining)
        else:  # concat: head-truncate
            content, _ = _truncate_to_tokens(content, remaining)
        fitted = {**m, "content": content}
        fitted_loop.append(fitted)
        loop_used += message_tokens(fitted)

    hist_budget = max(0, avail - loop_used)

    # 2) History into the remaining budget.
    summary_text: Optional[str] = None
    through = 0
    if strategy == "concat":
        _, kept, split_index = _concat_window_capped(history, hist_budget)
        window_ids = list(range(split_index, len(history)))
        base = [*kept, current_msg]
        dropped_turns = split_index
    else:  # window_summary
        might_summarize = len(history) > window_turns or \
            sum(message_tokens(m) for m in history) > hist_budget
        room = max(0, hist_budget - (SUMMARY_MAX_TOKENS if might_summarize else 0))
        older, window, split_index = _window_capped(history, window_turns, room)
        window_ids = list(range(split_index, len(history)))
        if older:
            summary_text, cost = await summarize_increment(
                None, older, api_key=api_key, model=model, base_url=base_url)
            summary_cost += cost
            # `through` = positions the summary actually covers. Only set it on
            # success: if the summary call fails/returns empty the `older` turns are
            # dropped without coverage, so the summary must NOT claim to cover them.
            if summary_text:
                through = split_index
        base = []
        if summary_text:
            base.append({"role": "system",
                         "content": f"以下是更早对话的摘要，供参考：\n{summary_text}"})
        base.extend(window)
        base.append(current_msg)
        # `older` turns left the verbatim window regardless of whether the summary
        # succeeded — report the drop honestly (a failed summary under tight budget
        # is real information loss, exactly what this phase measures).
        dropped_turns = split_index if older else 0

    return {
        "messages": [*base, *fitted_loop],
        "base_len": len(base),
        "strategy": strategy,
        "summary": summary_text,
        "summary_through_position": through,
        "included_history_ids": window_ids,
        "dropped_turns": dropped_turns,
        "summary_cost": summary_cost,
    }
