# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (always required before any pip or uvicorn commands)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Postgres (Docker container named myaibuddy-postgres)
docker start myaibuddy-postgres

# Run the dev server with auto-reload
uvicorn main:app --reload

# Open the app
open http://localhost:8000

# Interactive API docs
open http://localhost:8000/docs

# Run the context-management eval set (server must be running; see docs/eval_plan.md)
python eval/run.py --strategy concat --tier smoke    # fast 12-case set → eval/results/*.json
python eval/run.py --strategy concat --tier long     # 6-case set, ~8.7k-tok histories (chunking stress)
python eval/run.py --strategy concat --tier smoke --case ml_001   # single case
python eval/run.py --strategy concat --tier long --model grok-4-fast   # pick model under test
```

There are no tests and no linter configured yet.

## Design docs (in `docs/`)

- `docs/design.md` — full design document (architecture, backend, frontend, context-management roadmap)
- `docs/eval_plan.md` — context-management measurement & visualization plan (metrics, needle dataset, debugger Eval tab); implemented
- `docs/phase1_sliding_window_summary.md` — Phase 1 plan: sliding window + rolling summary
- `docs/phase2_pgvector_rag.md` — Phase 2 plan: pgvector RAG retrieval
- `docs/phase3_cross_session_memory.md` — Phase 3 plan: cross-session retrieval + memory injection
- `docs/reflections.md` — session reflections

## Architecture

**My AI Buddy** is a ChatGPT-like web app: a FastAPI backend that serves a single-page HTML frontend and proxies chat requests to the AI Builder Space API (`https://space.ai-builders.com/backend/v1`).

### File layout

| File | Role |
|---|---|
| `main.py` | All FastAPI routes, agentic loop, SSE streaming |
| `auth.py` | Google OAuth flow, JWT sign/verify, FastAPI dependencies |
| `database.py` | SQLAlchemy async engine setup, `get_db` session dependency, `init_db` |
| `models.py` | ORM models: `User`, `Chat`, `Message` |
| `context_report.py` | tiktoken-based token estimation + per-round `context_report` builder |
| `eval/run.py` | CLI eval runner (`--tier smoke\|long`, two-level judging; see `docs/eval_plan.md`) |
| `eval/cases/smoke/` | Frozen needle cases, smoke tier (12 cases, ~1.5k-tok histories — fast regression) |
| `eval/cases/long/` | Frozen needle cases, long tier (6 cases, ~8.7k-tok histories — compression/chunking stress) |
| `eval/results/` | Self-contained eval result JSONs (tagged `dataset_version`), rendered by the debugger Eval tab |
| `static/index.html` | Entire frontend (vanilla JS, no build step) |
| `static/debugger.html` | Context Debugger: Single Run view + Eval tab (standalone page) |

### Agentic loop (`POST /chat`)

Returns a **Server-Sent Events** stream. Each turn calls the upstream LLM with two tool schemas (`web_search`, `read_page`). If the model returns tool calls, the backend executes them and loops (up to `MAX_TURNS = 3`). On the final turn, or if max turns are exhausted, a `{"type":"done","reply":"..."}` event is emitted.

SSE event types: `status`, `done`, `error`.

### Authentication

Google OAuth 2.0 → JWT in an `HttpOnly + SameSite=Lax` cookie (`access_token`).

- `get_current_user` — returns `User | None` (guest-safe)
- `require_user` — raises 401 if unauthenticated; used on all `/api/*` routes

### Database

Tables are auto-created on startup via `init_db()` (no migration files needed for schema changes in dev — just drop and restart). For schema migrations in production use Alembic (`alembic` is installed).

```
users      id (uuid), google_id, email, name, picture, created_at
chats      id (uuid), user_id → users, title, model, created_at
messages   id (uuid), chat_id → chats, role, content, position
```

`Message.position` is the ordering key within a chat (no `created_at` on messages).

### API endpoints (chat management)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/chats` | required | List user's chats |
| `POST` | `/api/chats` | required | Create new chat |
| `PATCH` | `/api/chats/{id}` | required | Rename chat title |
| `DELETE` | `/api/chats/{id}` | required | Delete chat + all messages |
| `GET` | `/api/chats/{id}/messages` | required | Load messages for a chat |
| `POST` | `/api/chats/{id}/messages` | required | Append messages to a chat |
| `DELETE` | `/api/chats/{id}/messages/from/{position}` | required | Truncate messages from position N onwards (used by edit feature) |
| `POST` | `/api/chats/{id}/title` | required | LLM-generate a ≤16-char title; updates DB |
| `POST` | `/api/migrate` | required | Import localStorage chats into DB |

### Debug / Context Debugger endpoints (no auth required)

| Method | Path | Description |
|---|---|---|
| `GET` | `/debugger` | Serves `static/debugger.html` |
| `POST` | `/debug/run` | Runs full agentic loop; **SSE stream** — emits `context_report` (per-round token/layer breakdown) before each LLM call, `round` events carry API `usage`; accepts optional `strategy` (only `"concat"` for now); client abort stops backend |
| `POST` | `/debug/regenerate` | Accepts `{model, messages}`; calls LLM once (no tools); returns `{reply}` (also used by eval runner as LLM judge) |
| `GET` | `/debug/eval/results` | Lists result files in `eval/results/` |
| `GET` | `/debug/eval/results/{name}` | Serves one result file (consumed by the debugger Eval tab) |

Open the debugger at `http://localhost:8000/debugger`. It lets you inspect each agentic loop round, edit tool call arguments and results, disable rounds, insert synthetic rounds, and regenerate the final answer.

### Frontend features

- **Edit message**: hover a user bubble → pencil icon appears → click to edit inline. On save, messages from that position onwards are deleted from DB and the message is re-sent with truncated history as context.
- **LLM session titles**: after the first reply, `generateTitle()` fires async (non-blocking) to generate a ≤16-char summary via the LLM. Falls back to string truncation if the LLM call fails.
- **Session menu**: hover a sidebar item → ⋮ button → Rename (inline input) or Delete (removes from DB + sidebar).
- **Guest mode**: chats in `localStorage` (`aibc_chats_v1`); edit works in-memory only; titles are truncated (no LLM call). Migration to DB on first login.
- **Stop / abort**: while a chat call is in flight, the send button becomes a red ■ stop button. Clicking it aborts the fetch (`AbortController`); FastAPI stops the SSE generator; shows `"Response stopped."` in the thread. The unsaved user message is removed from `chat.messages` to avoid orphaned history.

### Debug logging

All backend log lines are gated by `DEBUG=true` in `.env` via a `log()` helper in `main.py`. Log prefixes: `[Auth]`, `[Chat]`, `[Edit]`, `[Title]`, `[Migrate]`, `[Agent]`, `[System]`.

### Environment variables (`.env`)

```
SUPER_MIND_API_KEY      # AI Builder Space API key
DATABASE_URL            # postgresql://... (auto-converted to asyncpg dialect)
JWT_SECRET              # 32-byte hex secret
JWT_EXPIRE_DAYS         # default 7
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI     # http://localhost:8000/auth/google/callback (dev)
DEBUG                   # set to "true" to enable backend log output (default: false)
```

Swapping `DATABASE_URL` to a Neon connection string is all that's needed to move to a hosted Postgres — no code changes required.
