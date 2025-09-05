from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from domains.auth.services.auth_service import get_session_id, get_user_info
from core.database import MongoDB
from domains.auth.routes.auth_routes import router as auth_router
from domains.users.routes.user_routes import router as users_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    MongoDB.connect()
    yield
    MongoDB.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 리액트 앱 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )
    
app.include_router(auth_router)
app.include_router(users_router)



# 아래부터는 채팅방 테스트입니다

import asyncio
from fastapi import WebSocket, WebSocketDisconnect

class ChatRoomManager:
    _sockets: Dict[str, WebSocket] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def connect(cls, ws: WebSocket, session_id: str):
        if not ws:
            return
        async with cls._lock:
            old_ws = cls._sockets.get(session_id)
            if old_ws:
                try:
                    await old_ws.close(code=1000)
                except:
                    pass

            await ws.accept()
            cls._sockets[session_id] = ws

    @classmethod
    async def disconnect(cls, session_id: str):
        async with cls._lock:
            ws = cls._sockets.pop(session_id, None)
        if ws:  # 락 해제 후 close (안 막히게)
            try:
                await ws.close()
            except:
                pass

    @classmethod
    async def broadcast(cls, message: str):
        async with cls._lock:
            sockets = list(cls._sockets.items())  # 복사본
        dc_users = []
        for session_id, ws in sockets:
            try:
                await ws.send_text(message)
            except Exception:
                dc_users.append(session_id)

        for session_id in dc_users:
            await cls.disconnect(session_id)


@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    session_id = await get_session_id(ws)
    user_info = await get_user_info(session_id=session_id)
    if not user_info:
        await ws.close(code=4001)
        return
    
    await ChatRoomManager.connect(ws=ws, session_id=session_id)
    
    try:
        while True:    
            data = await ws.receive_text()
            
            user_info = await get_user_info(session_id=session_id)
            
            if not user_info:
                await ChatRoomManager.disconnect(session_id=session_id)
                return
            
            await ChatRoomManager.broadcast(f"{user_info.username}: {data}")
    except WebSocketDisconnect:
        await ChatRoomManager.disconnect(session_id=session_id)
        
        
from fastapi.responses import HTMLResponse

@app.get("/chat")
async def chat_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat Test</title>
    </head>
    <body>
        <h1>WebSocket Chat Test</h1>
        <div>
            <input id="messageInput" type="text" placeholder="메시지를 입력하세요" />
            <button onclick="sendMessage()">보내기</button>
        </div>
        <ul id="messages"></ul>

        <script>
            const ws = new WebSocket(`ws://${location.host}/ws/chat`);

            ws.onopen = () => {
                console.log("✅ WebSocket 연결됨");
            };

            ws.onmessage = (event) => {
                const messages = document.getElementById("messages");
                const li = document.createElement("li");
                li.textContent = event.data;
                messages.appendChild(li);
            };

            ws.onclose = (event) => {
                console.log("❌ WebSocket 닫힘", event);
            };

            function sendMessage() {
                const input = document.getElementById("messageInput");
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(input.value);
                    input.value = "";
                } else {
                    alert("⚠️ WebSocket 연결이 닫혀서 메시지를 보낼 수 없습니다.");
                }
            }
        </script>
    </body>
    </html>
    """)
