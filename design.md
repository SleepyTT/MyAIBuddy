# My AI Buddy — Design Document

## Overview

A ChatGPT-like web application with a FastAPI backend, agentic LLM loop, Google OAuth authentication, and PostgreSQL chat history. The frontend is a single-page app served directly by the backend.

---

## Architecture

```
Browser (index.html)
    │
    ├── GET  /                  → serves static/index.html
    ├── GET  /auth/google       → redirects to Google OAuth
    ├── GET  /auth/google/callback
    ├── POST /auth/logout
    ├── GET  /auth/me
    │
    ├── GET/POST /api/chats
    ├── GET/POST /api/chats/{id}/messages
    ├── DELETE   /api/chats/{id}
    ├── POST     /api/migrate   → import localStorage → DB
    │
    └── POST /chat              → SSE stream (agentic loop)

FastAPI (main.py)
    ├── auth.py       — Google OAuth, JWT sign/verify
    ├── database.py   — SQLAlchemy async engine
    └── models.py     — ORM models

PostgreSQL (Docker)
    ├── users
    ├── chats
    └── messages
```

---

## Backend

### Stack
- **FastAPI** — web framework
- **SQLAlchemy (async) + asyncpg** — async ORM + PostgreSQL driver
- **python-jose** — JWT signing / verification
- **httpx** — async HTTP client (AI Builder API, Google OAuth, web fetch)
- **BeautifulSoup4** — HTML text extraction for `read_page` tool
- **python-dotenv** — `.env` loading

### Agentic Loop (`POST /chat`)

Returns a **Server-Sent Events (SSE)** stream so the frontend receives real-time status updates.

Flow per request:
1. Build message list from `history` + new user message
2. Loop up to `MAX_TURNS = 3`:
   - Call LLM with `[web_search, read_page]` tools, `tool_choice="auto"`
   - If no tool call → yield `{"type":"done","reply":"..."}` and exit
   - If `web_search` → yield `"Searching websites..."` status, execute, append tool result
   - If `read_page` → yield `"Reading webpage..."` status, execute, append tool result
3. If max turns exhausted → call LLM without tools, yield `done`

SSE event types:
| Event | Payload |
|---|---|
| `status` | `{"type":"status","message":"..."}` |
| `done` | `{"type":"done","reply":"..."}` |
| `error` | `{"type":"error","detail":"..."}` |

### Tools

| Tool | Description |
|---|---|
| `web_search(query)` | POST to AI Builder `/v1/search/`, returns top-3 results |
| `read_page(url)` | Fetches URL, strips HTML tags/scripts/nav, returns first 5000 chars |

### Authentication

- **Google OAuth 2.0** — standard authorization code flow
- **JWT** stored in `HttpOnly + SameSite=Lax` cookie (7-day expiry)
- `get_current_user` dependency → returns `User` or `None` (guest-friendly)
- `require_user` dependency → raises 401 if not authenticated

### Database (`models.py`)

```
users      id, google_id, email, name, picture, created_at
chats      id, user_id → users, title, model, created_at
messages   id, chat_id → chats, role, content, position, (no created_at — ordered by position)
```

Tables are auto-created on startup via `init_db()` (`Base.metadata.create_all`).

### New API Endpoints (added post-MVP)

| Method | Path | Description |
|---|---|---|
| `PATCH` | `/api/chats/{id}` | Rename a chat (updates `title` in DB) |
| `DELETE` | `/api/chats/{id}/messages/from/{position}` | Delete all messages at `position ≥ N` (used by edit feature) |
| `POST` | `/api/chats/{id}/title` | Call LLM to generate a ≤16-char title summary; updates DB and returns `{title}` |

### Context Debugger (`/debugger`)

A standalone debug tool for inspecting and replaying the agentic loop. No disk I/O — the trace is captured in memory during `/debug/run` and returned as structured JSON.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/debugger` | Serves `static/debugger.html` |
| `POST` | `/debug/run` | Runs full agentic loop; **SSE stream** — each round emitted as it completes; client abort stops backend |
| `POST` | `/debug/regenerate` | Accepts a modified `{model, messages}` array; calls LLM once without tools; returns `{reply}` |

**`/debug/run` SSE events (streamed as each round completes):**

| Event type | Payload | When |
|---|---|---|
| `status` | `{"type":"status","message":"Round N — calling LLM…"}` | Progress updates |
| `round` | `{"type":"round","data":{round, assistantMessage, toolResults, finalReply}}` | After each round finishes |
| `final` | `{"type":"final","reply":"..."}` | MAX_TURNS exhaustion only (no-tool final answer) |
| `done` | `{"type":"done"}` | Loop complete |
| `error` | `{"type":"error","detail":"..."}` | Upstream LLM failure |

The `round.data` fields:
- `assistantMessage` — raw message object from the LLM (may include `tool_calls`, `content`, or both)
- `toolResults` — list of `{id, name, arguments, content, status}` for each tool call in this round
- `finalReply` — non-null only when the LLM gave a direct answer (no tool calls) or MAX_TURNS exhaustion attaches the final to the last round

**Debugger UI (`static/debugger.html`):**
- Dark navy theme (`--page-bg: #060d17`)
- USER card — editable message; **REGENERATE TURN** re-runs the full loop from scratch
- AI round cards appear progressively as each round streams in; disable/enable toggle; **+ INSERT CARD** adds a synthetic round; inserted rounds can be removed
- Each AI card shows the assistant's response text (labeled **RESPONSE** for direct replies, **THINKING** for pre-tool content)
- TOOL CALL section — contenteditable JSON for argument editing
- TOOL RESULT section — textarea edit + active toggle (inactive results are excluded from regenerate)
- **■ STOP** button — while running, RUN becomes ■ STOP; clicking aborts the fetch (server stops on next `yield`); partial results already received are preserved in the UI
- **REGENERATE** button — calls `/debug/regenerate` with the reconstructed messages array and displays the new reply

**`buildMessages()` reconstruction rules:**
- Inactive rounds are skipped entirely
- Real rounds with no tool calls and no `finalReply` override are skipped (they were intermediate LLM responses without tools)
- Inserted synthetic rounds produce a minimal assistant message with matching `tool_call_id`s
- Real rounds deep-copy `assistantMessage`, filter out inactive tool calls, and apply edited argument values
- Each active tool result is appended as a `role: "tool"` message with matching `tool_call_id`

---

### Debug Logging

`main.py` has a `log()` helper that only prints when `DEBUG=true` in `.env`. Note: changing `.env` requires a full server restart — `--reload` only watches `.py` files.

| Tag | Meaning |
|---|---|
| `[Auth]` | Google OAuth flow: redirect, callback, JWT issuance, logout |
| `[Chat]` | Chat session CRUD: list, create, load messages, append messages, rename, delete |
| `[Edit]` | Message edit: DB truncation (delete messages from position N onwards) |
| `[Title]` | LLM-generated session title: input message, model used, result or fallback reason |
| `[Migrate]` | localStorage → DB migration on first login: chat count and message count per chat |
| `[Agent]` | Agentic loop: "LLM direct response, no tool is needed" if no tool call; "Round N \| Tool Call \| \<tool\> \| Succeed/Error" per tool call; "Max turns reached" if loop exhausted |

### Environment Variables (`.env`)

```
SUPER_MIND_API_KEY     — AI Builder Space API key
DATABASE_URL           — PostgreSQL connection string (swap for Neon, no code changes)
JWT_SECRET             — 32-byte hex secret
JWT_EXPIRE_DAYS        — default 7
GOOGLE_CLIENT_ID       — from Google Cloud Console
GOOGLE_CLIENT_SECRET   — from Google Cloud Console
GOOGLE_REDIRECT_URI    — http://localhost:8000/auth/google/callback
DEBUG                  — set to "true" to enable backend log output (default: false)
```

---

## Frontend (`static/index.html`)

Single HTML file with embedded CSS and JS. No build step, no framework.

### Layout
- **Left sidebar** (260px): logo, "+ New Chat" button, chat list grouped by date (Today / Yesterday / Past 7 days / Older), user profile / login at bottom
- **Main panel**: welcome screen → message thread → input bar with model selector

### Guest Mode
- Chats stored in `localStorage` (key: `aibc_chats_v1`)
- After 3 sent messages, a modal nudges the user to sign in
- On first login, all localStorage chats are POSTed to `/api/migrate` and localStorage is cleared (flag: `aibc_migrated`)
- Guest session titles: first message truncated to 16 chars (no LLM call)

### Edit Message
- Pencil icon appears at bottom-right of user message bubbles on hover (CSS-only show/hide)
- Click → bubble becomes an inline `<textarea>` with Cancel / Save & Send buttons
- **Save & Send always re-sends**, regardless of whether the text was changed — clicking Save is an explicit user intent to re-run from that point
- On save: all messages at `position ≥ edited_index` are deleted from DB (`DELETE /api/chats/{id}/messages/from/{position}`), the in-memory array is truncated, and `send(editedText)` is called with the truncated history as context
- For guest users: only in-memory array truncation + `lsSave()`; no DB call

### Session Sidebar
- Section header for same-day chats is **"Recents"** (not "Today")
- No icon prefix on chat items
- Hovering a chat item reveals a **⋮** button; clicking shows a dropdown with **Rename** and **Delete**
  - **Rename**: inline input replaces the title span; commits on Enter or blur; calls `PATCH /api/chats/{id}`
  - **Delete**: calls `DELETE /api/chats/{id}`, removes from in-memory `chats` array, resets to welcome screen if it was the active chat

### LLM Session Title Generation
- After the first message in a new chat receives a reply, `generateTitle()` fires asynchronously (fire-and-forget, does not block the UI)
- Calls `POST /api/chats/{id}/title` with the first user message and current model
- Backend prompts the LLM: *"用一句话总结以下内容，不超过16个字..."* using `deepseek-v4-flash` by default
- On LLM failure: falls back to truncating the message to 16 chars
- On re-edit of the first message: title is regenerated based on the new text

### Authenticated Mode
- Chat list loaded from `/api/chats` on boot
- Messages loaded lazily per chat from `/api/chats/{id}/messages`
- After each LLM reply, both the user message and assistant reply are saved to DB via `/api/chats/{id}/messages`

### Stop / Abort
- While an agentic call is in flight the send button switches to a **red ■ square** stop button
- Clicking it calls `AbortController.abort()`, which closes the SSE fetch; FastAPI detects the disconnection and stops the generator (no further LLM calls fire)
- On abort: thinking animation hides; a `"Response stopped."` hint is appended inline in the chat thread
- The unsaved user message is removed from `chat.messages` (it was never persisted to DB) so it doesn't appear as an orphaned user turn in the next send's history; the bubble stays visible in the DOM so the edit button still works

### Thinking Animation
- Three bouncing dots (CSS keyframe animation)
- Status message below the dots, updated in real time from SSE stream:
  - Default: `"Thinking..."`
  - On `web_search`: `"Searching websites for more information..."`
  - On `read_page`: `"Reading webpage..."`
  - On max-turns final call: `"Preparing answer..."`

### Libraries (CDN, no build step)
- `marked@12` — Markdown → HTML rendering
- `highlight.js 11.9` — syntax highlighting in code blocks (GitHub Dark theme)

### Model Selector
Supports all AI Builder Space models:
`supermind-agent-v1`, `grok-4-fast`, `gemini-2.5-pro`, `gemini-3-flash-preview`,
`deepseek-v4-flash`, `deepseek-v4-pro`, `deepseek`, `kimi-k2.5`, `gpt-5`

---

## Local Development

```bash
# Start Postgres
docker start myaibuddy-postgres

# Start server
source .venv/bin/activate && uvicorn main:app --reload

# Open app
open http://localhost:8000
```

### Moving to Neon (Production)
1. Create a Neon project and copy the connection string
2. In `.env`, replace `DATABASE_URL` with the Neon URL
3. Update `GOOGLE_REDIRECT_URI` to the production domain
4. Restart the server — no code changes needed
