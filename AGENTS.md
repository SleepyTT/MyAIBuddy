# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
```

There are no tests and no linter configured yet.

## Architecture

**My AI Buddy** is a ChatGPT-like web app: a FastAPI backend that serves a single-page HTML frontend and proxies chat requests to the AI Builder Space API (`https://space.ai-builders.com/backend/v1`).

### File layout

| File | Role |
|---|---|
| `main.py` | All FastAPI routes, agentic loop, SSE streaming |
| `auth.py` | Google OAuth flow, JWT sign/verify, FastAPI dependencies |
| `database.py` | SQLAlchemy async engine setup, `get_db` session dependency, `init_db` |
| `models.py` | ORM models: `User`, `Chat`, `Message` |
| `static/index.html` | Entire frontend (vanilla JS, no build step) |

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

### Frontend features

- **Edit message**: hover a user bubble → pencil icon appears → click to edit inline. On save, messages from that position onwards are deleted from DB and the message is re-sent with truncated history as context.
- **LLM session titles**: after the first reply, `generateTitle()` fires async (non-blocking) to generate a ≤16-char summary via the LLM. Falls back to string truncation if the LLM call fails.
- **Session menu**: hover a sidebar item → ⋮ button → Rename (inline input) or Delete (removes from DB + sidebar).
- **Guest mode**: chats in `localStorage` (`aibc_chats_v1`); edit works in-memory only; titles are truncated (no LLM call). Migration to DB on first login.

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
