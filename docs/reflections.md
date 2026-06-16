# Reflections — Learnings & Mistakes

A running log of bugs, surprises, and decisions made during development. Reference this before making changes to avoid repeating mistakes.

---

## Python & Environment

### `load_dotenv()` must be called in every module that reads env vars at import time
**What happened:** `database.py` reads `DATABASE_URL` at module level to create the SQLAlchemy engine. `main.py` called `load_dotenv()` but only *after* importing `database`, so the env var was empty when the engine was created — causing a startup crash.

**Fix:** Call `load_dotenv()` at the top of `database.py` (and any other module that reads env at import time), not just in `main.py`.

**Rule:** If a module reads `os.getenv(...)` at the top level (outside a function), it must call `load_dotenv()` itself.

---

### Python 3.9 does not support `X | None` union syntax
**What happened:** `.venv` uses Python 3.9. Writing `str | None` or `list[dict]` as type hints caused a `TypeError` at startup.

**Fix:** Use `Optional[str]` and `List[dict]` from `typing` for Python 3.9 compatibility.

**Rule:** Always use `typing.Optional` and `typing.List` until the project upgrades to Python 3.10+.

---

### Always install pip packages inside `.venv`
**What happened:** `beautifulsoup4` was installed into the system Python instead of the project venv, causing import errors when the server ran from `.venv`.

**Fix:** Always `source .venv/bin/activate` before any `pip install`. If `.venv` doesn't exist, create it first with `python3 -m venv .venv`.

---

### zsh treats square brackets as glob patterns
**What happened:** Running `pip install sqlalchemy[asyncio]` in zsh failed with `no matches found: sqlalchemy[asyncio]`.

**Fix:** Always quote extras in zsh: `pip install "sqlalchemy[asyncio]"`.

---

## Google / OAuth

### Google profile images require `referrerpolicy="no-referrer"`
**What happened:** The user's Google profile picture failed to load in the sidebar after login. Google's CDN (`lh3.googleusercontent.com`) rejects requests that include a `Referer` header pointing to a different origin.

**Fix:** Add `referrerpolicy="no-referrer"` to the `<img>` tag.

```html
<img id="user-avatar" src="" alt="" referrerpolicy="no-referrer">
```

---

## Infrastructure

### Docker is better than Homebrew for local Postgres
**What happened:** Installing Postgres via Homebrew required `sudo` access, which the user didn't have. Docker requires no system-level permissions beyond Docker Desktop being installed.

**Additional benefit:** The Docker connection string format (`postgresql://user:pass@localhost:5432/db`) is identical to Neon's, so moving to production is a one-line `.env` change.

**Rule:** Use Docker for all local backing services.

---

## LLM / Agentic Loop

### `supermind-agent-v1` ignores `tool_choice: "required"` — it has built-in search
**What happened:** When testing whether the LLM would call `web_search`, `supermind-agent-v1` answered directly without calling the tool — because it has its own internal web search capability.

**Learning:** For testing explicit tool-call behavior, use `grok-4-fast`. It reliably honors `tool_choice: "required"` and produces valid tool call JSON.

---

### Tool call results must use `role: "tool"` with matching `tool_call_id`
**Learning:** When appending tool results to the message list, the message must have `role: "tool"` and include the `tool_call_id` from the original tool call. Missing this causes the LLM to misinterpret the conversation history.

---

## Frontend

### SSE requires stream parsing, not `res.json()`
**What happened (anticipated):** When `/chat` was changed from returning JSON to SSE (`text/event-stream`), the frontend `fetch` call needed to switch from `await res.json()` to reading the response as a `ReadableStream` and parsing `data: {...}\n\n` chunks manually.

**Pattern:**
```js
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const parts = buffer.split('\n\n');
  buffer = parts.pop();
  for (const part of parts) {
    if (!part.startsWith('data: ')) continue;
    const event = JSON.parse(part.slice(6));
    // handle event
  }
}
```

---

### `EventSource` does not support POST requests
**Learning:** The browser's native `EventSource` API only works with GET. Since `/chat` is a POST endpoint, SSE must be consumed via `fetch` + `ReadableStream` (see pattern above).

---

## Tooling

### `sed` replaces strings inside function definitions too
**What happened:** Used `sed` to replace all `print(` calls with `log(` across `main.py`. The `sed` command also replaced the `print(` inside the `log()` helper function itself, turning it into `log(*args, **kwargs)` — an infinite recursive call.

**Fix:** Always manually verify the function definition after bulk replacements, or use a narrower pattern that excludes the helper's own body.

**Rule:** After any `sed` bulk-replace on a file, grep for the replaced pattern and visually inspect every hit before running the code.

---

## Frontend

### Inline edit requires index tracking from the moment bubbles are rendered
**Learning:** The edit feature needs to know each message's position in `chat.messages` at render time (to know how many messages to truncate). The index must be passed into `addBubble()` as a parameter and stored in the event closure. Trying to derive the index at click time (e.g., by counting DOM siblings) is fragile — assistant messages don't have edit buttons, so DOM count doesn't map 1:1 to array index.

**Pattern:** Pass `msgIndex` explicitly in `addBubble(role, content, msgIndex)` and use it directly in the edit button's event listener closure.

### Fire-and-forget async operations should not block UI
**Learning:** LLM title generation after first message is a nice-to-have that should not delay the chat response. Calling it without `await` in the frontend (`generateTitle(...)` without `await`) lets the sidebar update quietly in the background while the user can already type the next message.

### Dropdown positioning with `getBoundingClientRect`
**Learning:** Positioning a dropdown relative to a button inside a scrollable sidebar using `position: fixed` + `getBoundingClientRect()` is more reliable than `position: absolute` inside the scrollable container, because absolute positioning is affected by `overflow: hidden` on ancestor elements.

### `AbortController` pattern for cancelling SSE fetches
**Learning:** Pass `signal: controller.signal` to `fetch()`. When the user clicks stop, call `controller.abort()`. The in-flight `reader.read()` immediately rejects with an `AbortError`. Always catch `e.name === 'AbortError'` as a distinct branch — it's not an error, it's a user action, so show a "stopped" hint rather than an error message.

### "Save & Send" should always re-send — don't guard on unchanged text
**Learning:** An earlier version of `startEdit` had `if (newText === original) { cancelBtn.click(); return; }` to avoid re-sending unchanged messages. This broke the abort+re-edit flow: after aborting, `chat.messages.pop()` removes the user message, but the bubble stays visible. The user clicks edit → save, and since the text is unchanged, the guard silently cancels instead of re-sending. The fix: remove the guard entirely. Clicking "Save & Send" is an explicit user action — always re-send, whether or not the text changed.

### Pop the unsaved user message from history on abort
**Learning:** `send()` pushes the user message to `chat.messages` before the fetch, so it's available as context. If the user aborts before a reply arrives, that message was never persisted to DB. Leaving it in the array causes the next send's history to end with a user message and no assistant response, which can confuse the LLM. Fix: `chat.messages.pop()` inside the `AbortError` catch branch. The DOM bubble is already rendered and stays visible, so the user can still use the edit button to re-send.

---

## Context Debugger

### No disk I/O needed for LLM trace visualization
**Learning:** The agentic loop trace doesn't need to be written to a log file. Running the loop inside a dedicated `/debug/run` endpoint that captures each round into a list and returns structured JSON is simpler and more reliable — no file paths, no log parsing, no rotation concerns, and the full trace is available immediately in the browser as a JS object.

### SSE streaming enables progressive UI AND true server-side cancellation simultaneously
**Learning:** Converting `/debug/run` from a single JSON response to SSE yields two benefits at once: (1) rounds appear in the UI as they complete rather than all at once after a long wait, and (2) when the client aborts the fetch via `AbortController`, the FastAPI async generator detects the disconnection on the next `yield` and halts — no further LLM calls are made. Streaming is not just a UX nicety; it's what makes abort actually stop the backend.

### Reconstructing messages for regenerate requires matching `tool_call_id`s
**Learning:** When the user edits tool call arguments or disables tool results in the debugger and hits REGENERATE, the backend needs to receive a well-formed messages array. The assistant's message must list only the active tool calls (with edited arguments), and each `role: "tool"` result message must have a `tool_call_id` that matches exactly. Synthetic/inserted rounds need fake but consistent IDs generated at insert time — the same ID must appear in the assistant's `tool_calls[].id` and in the tool result's `tool_call_id`.

### `contenteditable` for JSON editing in the browser
**Learning:** Using `contenteditable="true"` on a `<pre>` element lets users edit JSON inline without a full editor library. Committing on `blur` (not on every keystroke) prevents mid-edit parses. Always catch `JSON.parse` errors and show a visible error state rather than silently discarding the edit.

### `for...else` is the cleanest Python pattern for MAX_TURNS exhaustion
**Learning:** Using `for turn in range(MAX_TURNS): ... else: <exhausted>` (where `else` runs only if the loop wasn't `break`ed) is the most readable way to distinguish "loop ended naturally" from "loop exited early with a reply." Avoids a separate boolean flag.
