# My AI Buddy

**Version 1.0**

A ChatGPT-like web application built with a FastAPI backend and a vanilla JS single-page frontend. Supports real-time streaming responses, an agentic LLM loop with web search and page reading tools, Google OAuth authentication, and persistent chat history in PostgreSQL.

---

## Features

### Chat
- **Agentic LLM loop** — the model can autonomously call `web_search` and `read_page` tools (up to 3 rounds) before returning a final answer
- **Real-time streaming** — responses stream via Server-Sent Events (SSE); status messages update live during tool calls ("Searching websites…", "Reading webpage…")
- **Stop / abort** — while a response is in flight, the send button becomes a red ■ stop button; clicking it cancels both the frontend fetch and the backend SSE generator
- **Edit message** — hover any user bubble to reveal a pencil icon; click to edit inline; "Save & Send" always re-sends from that point, deleting subsequent messages and re-running the LLM
- **Model selector** — switch between all supported AI Builder Space models per conversation

### Sessions
- **Persistent chat history** — authenticated users' chats and messages are stored in PostgreSQL and loaded on demand
- **LLM-generated titles** — after the first reply, a short (≤16-char) summary title is generated asynchronously and shown in the sidebar
- **Rename & delete** — hover a sidebar item → ⋮ menu → rename inline or delete the session (removes from DB and sidebar)
- **Guest mode** — chats stored in `localStorage`; on first login all local chats are migrated to the DB automatically

### Authentication
- **Google OAuth 2.0** — sign in with Google; JWT stored in an `HttpOnly + SameSite=Lax` cookie (7-day expiry)
- **Guest-friendly** — full chat works without login; a nudge appears after 3 messages

### Context Debugger (`/debugger`)
A standalone developer tool for inspecting and replaying the agentic loop:
- Rounds stream into the UI one by one as they complete
- Each round shows the assistant's response text, tool call arguments (editable JSON), and tool results (editable text)
- Disable/enable individual rounds or tool results
- Insert synthetic tool call/result cards
- **Regenerate** — reconstructs the modified message context and calls the LLM again to see how the answer changes
- **■ STOP** button — abort a running debug session mid-way

---

## File Structure

| File / Folder | Description |
|---|---|
| `main.py` | All FastAPI routes: auth, chat CRUD, agentic loop (`POST /chat`), debug endpoints (`/debug/run`, `/debug/regenerate`) |
| `auth.py` | Google OAuth 2.0 flow, JWT signing / verification, `get_current_user` and `require_user` FastAPI dependencies |
| `database.py` | SQLAlchemy async engine setup, `get_db` session dependency, `init_db` (auto-creates tables on startup) |
| `models.py` | ORM models: `User`, `Chat`, `Message` |
| `requirements.txt` | Python dependencies |
| `static/index.html` | Entire frontend — vanilla JS, embedded CSS, no build step |
| `static/debugger.html` | Context Debugger tool — standalone dark-theme page, no backend auth required |
| `design.md` | Architecture and feature design document |
| `reflections.md` | Running log of bugs, surprises, and lessons learned during development |
| `CLAUDE.md` | Guidance for Claude Code: commands, architecture overview, API reference |
| `AGENTS.md` | Agent configuration notes |
| `AI-Builder-Space_OpenAPI_Json.json` | OpenAPI spec for the AI Builder Space upstream API |
| `.gitignore` | Excludes `.env`, `.venv/`, `__pycache__/`, `.claude/` |

---

## Local Development

### Prerequisites
- Python 3.9+
- Docker Desktop (for local PostgreSQL)
- A [Google Cloud Console](https://console.cloud.google.com/) OAuth 2.0 client
- An AI Builder Space API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/SleepyTT/MyAIBuddy.git
cd MyAIBuddy

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env (see template below)
cp .env.example .env   # or create manually

# 5. Start local PostgreSQL
docker run -d --name myaibuddy-postgres \
  -e POSTGRES_USER=myaibuddy \
  -e POSTGRES_PASSWORD=<your_password> \
  -e POSTGRES_DB=myaibuddy \
  -p 5432:5432 postgres:15

# 6. Start the dev server
uvicorn main:app --reload

# 7. Open the app
open http://localhost:8000
```

### Environment Variables (`.env`)

```
SUPER_MIND_API_KEY=      # AI Builder Space API key
DATABASE_URL=            # postgresql://user:pass@localhost:5432/myaibuddy
JWT_SECRET=              # 32-byte hex string (e.g. openssl rand -hex 32)
JWT_EXPIRE_DAYS=7
GOOGLE_CLIENT_ID=        # from Google Cloud Console
GOOGLE_CLIENT_SECRET=    # from Google Cloud Console
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
DEBUG=false              # set to "true" to enable backend log output
```

### Supported Models

`supermind-agent-v1` · `grok-4-fast` · `gemini-2.5-pro` · `gemini-3-flash-preview` · `deepseek-v4-flash` · `deepseek-v4-pro` · `deepseek` · `kimi-k2.5` · `gpt-5`

---

## Moving to Production (Neon + hosted deployment)

1. Create a [Neon](https://neon.tech) project and copy the connection string
2. In `.env`, set `DATABASE_URL` to the Neon URL — no code changes required
3. Update `GOOGLE_REDIRECT_URI` to your production domain
4. Restart the server
