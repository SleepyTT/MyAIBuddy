import json
import os
from typing import Annotated, Any, List, Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from auth import (
    create_jwt, get_current_user, require_user,
    google_auth_url, exchange_code_for_user_info, get_or_create_user,
)
from database import get_db, init_db
from models import Chat, Message, User

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

AI_BUILDER_BASE_URL = "https://space.ai-builders.com/backend/v1"

# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current or real-time information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
}


async def web_search(query: str) -> dict[str, Any]:
    api_key = os.getenv("SUPER_MIND_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"keywords": [query], "max_results": 3}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{AI_BUILDER_BASE_URL}/search/",
            json=payload,
            headers=headers,
        )
    resp.raise_for_status()
    return resp.json()


READ_PAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": "Fetch a web page and extract its main text content. Use this to read the details of a specific URL found via web_search.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch and read.",
                }
            },
            "required": ["url"],
        },
    },
}

PAGE_TEXT_LIMIT = 5_000  # chars sent to the LLM


async def read_page(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    text = "\n".join(lines)
    return text[:PAGE_TEXT_LIMIT]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

tags_metadata = [
    {
        "name": "greeting",
        "description": "Endpoints that return a greeting string based on your input.",
    },
    {
        "name": "chat",
        "description": "Proxy chat endpoint forwarding to the AI Builder Space.",
    },
]

app = FastAPI(
    title="My AI Buddy",
    version="0.1.0",
    description="My AI Buddy API. Open **/docs** to try requests from Swagger UI.",
    openapi_tags=tags_metadata,
)


@app.on_event("startup")
async def startup():
    await init_db()


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/auth/google", include_in_schema=False)
def auth_google():
    log("[Auth] Redirecting to Google OAuth")
    return RedirectResponse(google_auth_url())


@app.get("/auth/google/callback", include_in_schema=False)
async def auth_google_callback(
    code: str,
    db: AsyncSession = Depends(get_db),
):
    log("[Auth] Google callback received, exchanging code for user info...")
    user_info = await exchange_code_for_user_info(code)
    log(f"[Auth] Got user info: email={user_info.get('email')} name={user_info.get('name')}")
    user = await get_or_create_user(user_info, db)
    log(f"[Auth] User {'created' if not user.created_at else 'loaded'}: id={user.id} email={user.email}")
    token = create_jwt(user.id)
    response = RedirectResponse(url="/")
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * int(os.getenv("JWT_EXPIRE_DAYS", "7")),
    )
    log(f"[Auth] JWT issued for user {user.id}, redirecting to /")
    return response


@app.post("/auth/logout", include_in_schema=False)
def auth_logout():
    log("[Auth] User logged out, clearing cookie")
    response = JSONResponse({"ok": True})
    response.delete_cookie("access_token")
    return response


@app.get("/auth/me", include_in_schema=False)
async def auth_me(user: Optional[User] = Depends(get_current_user)):
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse({
        "authenticated": True,
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "picture": user.picture,
    })


# ---------------------------------------------------------------------------
# Chat history CRUD (authenticated)
# ---------------------------------------------------------------------------

@app.get("/api/chats", include_in_schema=False)
async def list_chats(
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Chat).where(Chat.user_id == user.id).order_by(Chat.created_at.desc())
    )
    chats = result.scalars().all()
    log(f"[Chat] List chats for user={user.email}: {len(chats)} session(s)")
    return JSONResponse([{
        "id": c.id, "title": c.title, "model": c.model,
        "createdAt": c.created_at.timestamp() * 1000,
    } for c in chats])


@app.post("/api/chats", include_in_schema=False)
async def create_chat(
    body: dict,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    chat = Chat(
        user_id=user.id,
        title=body.get("title", "New Chat"),
        model=body.get("model", "supermind-agent-v1"),
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    log(f"[Chat] Created new chat: id={chat.id} title='{chat.title}' model={chat.model} user={user.email}")
    return JSONResponse({"id": chat.id, "title": chat.title, "model": chat.model, "createdAt": chat.created_at.timestamp() * 1000})


@app.get("/api/chats/{chat_id}/messages", include_in_schema=False)
async def get_messages(
    chat_id: str,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.position)
    )
    msgs = result.scalars().all()
    log(f"[Chat] Load messages: chat_id={chat_id} count={len(msgs)}")
    return JSONResponse([{"role": m.role, "content": m.content} for m in msgs])


@app.post("/api/chats/{chat_id}/messages", include_in_schema=False)
async def append_messages(
    chat_id: str,
    body: dict,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get current max position
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.position.desc())
    )
    last = result.scalars().first()
    pos = (last.position + 1) if last else 0

    new_msgs = body.get("messages", [])
    for msg in new_msgs:
        db.add(Message(chat_id=chat_id, role=msg["role"], content=msg["content"], position=pos))
        pos += 1

    if body.get("title"):
        chat.title = body["title"]

    await db.commit()
    log(f"[Chat] Appended {len(new_msgs)} message(s) to chat_id={chat_id} (positions up to {pos - 1})")

    return JSONResponse({"ok": True})


@app.delete("/api/chats/{chat_id}", include_in_schema=False)
async def delete_chat(
    chat_id: str,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    log(f"[Chat] Deleted chat: id={chat_id} title='{chat.title}' user={user.email}")
    await db.delete(chat)
    await db.commit()
    return JSONResponse({"ok": True})


@app.patch("/api/chats/{chat_id}", include_in_schema=False)
async def update_chat(
    chat_id: str,
    body: dict,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if "title" in body:
        old_title = chat.title
        chat.title = body["title"]
        log(f"[Chat] Renamed chat_id={chat_id}: '{old_title}' → '{chat.title}'")
    await db.commit()
    return JSONResponse({"ok": True})


@app.delete("/api/chats/{chat_id}/messages/from/{position}", include_in_schema=False)
async def delete_messages_from_position(
    chat_id: str,
    position: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    log(f"[Edit] Truncating chat_id={chat_id}: deleting messages from position {position} onwards")
    await db.execute(
        delete(Message).where(Message.chat_id == chat_id, Message.position >= position)
    )
    await db.commit()
    log(f"[Edit] Truncation done for chat_id={chat_id}")
    return JSONResponse({"ok": True})


@app.post("/api/chats/{chat_id}/title", include_in_schema=False)
async def generate_chat_title(
    chat_id: str,
    body: dict,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id))
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    message = body.get("message", "")
    model = body.get("model", "deepseek-v4-flash")
    title = message[:16] + ("…" if len(message) > 16 else "")  # fallback
    log(f"[Title] Generating title for chat_id={chat_id} using model={model}")
    log(f"[Title] First message: '{message[:80]}{'...' if len(message) > 80 else ''}'")

    api_key = os.getenv("SUPER_MIND_API_KEY")
    if api_key:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"用一句话总结以下内容，不超过16个字，只返回总结文字，不要任何标点符号和额外解释：{message}"}],
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{AI_BUILDER_BASE_URL}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"].get("content", "").strip()
                if raw:
                    title = raw[:16] + ("…" if len(raw) > 16 else "")
                    log(f"[Title] LLM generated: '{title}'")
                else:
                    log(f"[Title] LLM returned empty, using fallback: '{title}'")
            else:
                log(f"[Title] LLM call failed (status {resp.status_code}), using fallback: '{title}'")
        except Exception as e:
            log(f"[Title] LLM call exception: {e}, using fallback: '{title}'")
    else:
        log(f"[Title] No API key, using fallback: '{title}'")

    chat.title = title
    await db.commit()
    return JSONResponse({"title": title})


@app.post("/api/migrate", include_in_schema=False)
async def migrate_local_chats(
    body: dict,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Accepts localStorage chats array and imports them into the DB."""
    local_chats = body.get("chats", [])
    log(f"[Migrate] Importing {len(local_chats)} local chat(s) for user={user.email}")
    imported = 0
    for lc in local_chats:
        chat = Chat(
            user_id=user.id,
            title=lc.get("title", "Imported Chat"),
            model=lc.get("model", "supermind-agent-v1"),
        )
        db.add(chat)
        await db.flush()  # get chat.id

        msg_count = 0
        for pos, msg in enumerate(lc.get("messages", [])):
            if msg.get("role") in ("user", "assistant"):
                db.add(Message(chat_id=chat.id, role=msg["role"], content=msg["content"], position=pos))
                msg_count += 1
        log(f"[Migrate]   → chat '{chat.title}': {msg_count} message(s)")
        imported += 1

    await db.commit()
    log(f"[Migrate] Done. Imported {imported} chat(s).")
    return JSONResponse({"imported": imported})


class HelloResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "hello：Alice"}},
    )

    message: str = Field(
        ...,
        description='Greeting text: the literal prefix "hello：" followed by your `name` query value.',
    )


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to send to the AI coach.")
    model: str = Field(
        ...,
        description=(
            "Model to use for the completion. Supported values: "
            "`deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `supermind-agent-v1`, "
            "`kimi-k2.5`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gpt-5`, `grok-4-fast`."
        ),
    )
    history: Optional[List[dict]] = Field(default_factory=list, description="Previous user/assistant messages for context.")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The AI coach's response.")


MAX_TURNS = 3


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def status(message: str) -> str:
    return sse({"type": "status", "message": message})


@app.post("/chat", tags=["chat"], summary="Chat with My AI Buddy (SSE stream)")
async def chat(body: ChatRequest) -> StreamingResponse:
    async def generate():
        api_key = os.getenv("SUPER_MIND_API_KEY")
        if not api_key:
            yield sse({"type": "error", "detail": "SUPER_MIND_API_KEY not configured"})
            return

        headers = {"Authorization": f"Bearer {api_key}"}
        messages: List[dict[str, Any]] = [*(body.history or []), {"role": "user", "content": body.message}]

        yield status("Thinking...")

        for turn in range(MAX_TURNS):
            payload = {
                "model": body.model,
                "messages": messages,
                "tools": [WEB_SEARCH_TOOL, READ_PAGE_TOOL],
                "tool_choice": "auto",
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{AI_BUILDER_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )

            if resp.status_code != 200:
                yield sse({"type": "error", "detail": resp.text})
                return

            assistant_message = resp.json()["choices"][0]["message"]
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []

            if not tool_calls:
                log(f"[Agent] LLM direct response, no tool is needed")
                reply = assistant_message.get("content") or ""
                yield sse({"type": "done", "reply": reply})
                return

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                fn_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]

                log(f"[Agent] Decided to call tool.")

                if fn_name == "web_search":
                    yield status("Searching websites for more information...")
                    try:
                        result = await web_search(fn_args["query"])
                        result_str = json.dumps(result, ensure_ascii=False)
                        log(f"[Agent] Round {turn + 1} | Tool Call | {fn_name} | Succeed")
                    except Exception as e:
                        result_str = f"Error: {e}"
                        log(f"[Agent] Round {turn + 1} | Tool Call | {fn_name} | Error")
                elif fn_name == "read_page":
                    yield status("Reading webpage...")
                    try:
                        result_str = await read_page(fn_args["url"])
                        log(f"[Agent] Round {turn + 1} | Tool Call | {fn_name} | Succeed")
                    except Exception as e:
                        result_str = f"Error: {e}"
                        log(f"[Agent] Round {turn + 1} | Tool Call | {fn_name} | Error")
                else:
                    result_str = f"Unknown tool: {fn_name}"
                    log(f"[Agent] Round {turn + 1} | Tool Call | {fn_name} | Error")

                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result_str})

        # Max turns exhausted — force a final answer without tools
        log(f"[Agent] Max turns ({MAX_TURNS}) reached, requesting final answer without tools")
        yield status("Preparing answer...")
        payload = {"model": body.model, "messages": messages}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{AI_BUILDER_BASE_URL}/chat/completions", json=payload, headers=headers)
        if resp.status_code != 200:
            yield sse({"type": "error", "detail": resp.text})
            return

        reply = resp.json()["choices"][0]["message"].get("content") or ""
        yield sse({"type": "done", "reply": reply})

    return StreamingResponse(generate(), media_type="text/event-stream")


class ToolCallTestResponse(BaseModel):
    tool_called: bool = Field(..., description="Whether the LLM produced a tool call.")
    tool_name: Optional[str] = Field(None, description="Name of the tool the LLM chose to call.")
    tool_arguments: Optional[dict[str, Any]] = Field(None, description="Arguments the LLM passed to the tool.")
    raw_message: dict[str, Any] = Field(..., description="Full assistant message returned by the LLM.")


@app.post(
    "/chat/tool-call-test",
    tags=["chat"],
    summary="Verify the LLM produces a tool call for a web-search question",
    description=(
        "Sends a hardcoded question ('Who won the Super Bowl?') to the LLM together with "
        "the `web_search` tool schema. Returns whether the model responded with a valid "
        "tool call. **Does not execute the tool** — this is a schema verification only."
    ),
    response_model=ToolCallTestResponse,
)
async def tool_call_test(model: str = Query(..., description="Model to use for the test.")) -> ToolCallTestResponse:
    api_key = os.getenv("SUPER_MIND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SUPER_MIND_API_KEY not configured")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Who won the Super Bowl?"}],
        "tools": [WEB_SEARCH_TOOL],
        "tool_choice": "required",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{AI_BUILDER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    assistant_message = resp.json()["choices"][0]["message"]
    tool_calls = assistant_message.get("tool_calls") or []

    if tool_calls:
        first = tool_calls[0]
        return ToolCallTestResponse(
            tool_called=True,
            tool_name=first["function"]["name"],
            tool_arguments=json.loads(first["function"]["arguments"]),
            raw_message=assistant_message,
        )

    return ToolCallTestResponse(
        tool_called=False,
        tool_name=None,
        tool_arguments=None,
        raw_message=assistant_message,
    )


@app.get(
    "/hello",
    tags=["greeting"],
    summary="Say hello by name",
    description=(
        "Send a **GET** request with required query parameter **`name`** (non-empty string). "
        "Example: `/hello?name=Alice`. Returns JSON `{ \"message\": \"hello：<name>\" }`."
    ),
    response_model=HelloResponse,
    response_description="JSON object containing the greeting in `message`.",
)
def hello(
    name: Annotated[
        str,
        Query(
            min_length=1,
            description="Name to greet; passed as a URL query parameter.",
            examples={
                "english": {"summary": "ASCII name", "value": "Alice"},
                "unicode": {"summary": "Unicode name", "value": "小明"},
            },
        ),
    ],
) -> HelloResponse:
    return HelloResponse(message=f"hello：{name}")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


@app.get("/debugger", include_in_schema=False)
def debugger_page():
    return FileResponse("static/debugger.html")


# ---------------------------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------------------------

@app.post("/debug/run", include_in_schema=False)
async def debug_run(body: dict):
    api_key = os.getenv("SUPER_MIND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SUPER_MIND_API_KEY not configured")

    model = body.get("model", "supermind-agent-v1")
    message = body.get("message", "")
    history = body.get("history", [])

    async def generate():
        messages: List[dict[str, Any]] = [*history, {"role": "user", "content": message}]

        for round_num in range(1, MAX_TURNS + 1):
            yield sse({"type": "status", "message": f"Round {round_num} — calling LLM…"})

            payload = {
                "model": model,
                "messages": messages,
                "tools": [WEB_SEARCH_TOOL, READ_PAGE_TOOL],
                "tool_choice": "auto",
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{AI_BUILDER_BASE_URL}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code != 200:
                yield sse({"type": "error", "detail": resp.text})
                return

            assistant_message = resp.json()["choices"][0]["message"]
            messages.append(assistant_message)
            tool_calls_raw = assistant_message.get("tool_calls") or []

            if not tool_calls_raw:
                yield sse({"type": "round", "data": {
                    "round": round_num,
                    "assistantMessage": assistant_message,
                    "toolResults": [],
                    "finalReply": assistant_message.get("content") or "",
                }})
                break

            tool_results = []
            for tc in tool_calls_raw:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    fn_args = {}
                tc_id = tc["id"]

                if fn_name == "web_search":
                    yield sse({"type": "status", "message": f"Round {round_num} — searching web…"})
                elif fn_name == "read_page":
                    yield sse({"type": "status", "message": f"Round {round_num} — reading page…"})

                try:
                    if fn_name == "web_search":
                        result = await web_search(fn_args.get("query", ""))
                        result_str = json.dumps(result, ensure_ascii=False)
                        tool_status = "succeed"
                    elif fn_name == "read_page":
                        result_str = await read_page(fn_args.get("url", ""))
                        tool_status = "succeed"
                    else:
                        result_str = f"Unknown tool: {fn_name}"
                        tool_status = "error"
                except Exception as e:
                    result_str = str(e)
                    tool_status = "error"

                tool_results.append({
                    "id": tc_id,
                    "name": fn_name,
                    "arguments": fn_args,
                    "content": result_str,
                    "status": tool_status,
                })
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_str})

            yield sse({"type": "round", "data": {
                "round": round_num,
                "assistantMessage": assistant_message,
                "toolResults": tool_results,
                "finalReply": None,
            }})
        else:
            # MAX_TURNS exhausted — force final answer without tools
            yield sse({"type": "status", "message": "Max turns reached — getting final answer…"})
            payload = {"model": model, "messages": messages}
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{AI_BUILDER_BASE_URL}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            final = resp.json()["choices"][0]["message"].get("content") or "" if resp.status_code == 200 else ""
            yield sse({"type": "final", "reply": final})

        yield sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/debug/regenerate", include_in_schema=False)
async def debug_regenerate(body: dict):
    api_key = os.getenv("SUPER_MIND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SUPER_MIND_API_KEY not configured")

    model = body.get("model", "supermind-agent-v1")
    messages = body.get("messages", [])

    payload = {"model": model, "messages": messages}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{AI_BUILDER_BASE_URL}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)

    reply = resp.json()["choices"][0]["message"].get("content") or ""
    return JSONResponse({"reply": reply})


app.mount("/static", StaticFiles(directory="static"), name="static")
