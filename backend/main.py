from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import json
import asyncio
from modules.logger import log
from typing import Dict
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from domains.auth.services.auth_service import get_session_id, get_user_info, refresh_session
from modules.database import MongoDB, Redis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from domains.auth.routes.auth_routes import router as auth_router
from domains.users.routes.user_routes import router as users_router

FLUSH_INTERVAL = 10  # seconds

# -------------------------------
# Redis 셋팅
# -------------------------------
MAX_MSG = 100  # Redis 캐시 최대 메시지 수
CHAT_REDIS_KEY = "chat"

# -------------------------------
# ChatRoomManager
# -------------------------------
class ChatRoomManager:
    sockets: Dict[str, WebSocket] = {}

    @classmethod
    async def connect(cls, ws: WebSocket, uid: str):
        await ws.accept()

        old_ws = cls.sockets.get(uid, None)

        if old_ws:
            old_ws.close()

        cls.sockets[uid] = ws

        # Redis에서 최근 메시지 불러오기
        redis = Redis.get_client()
        recent_msgs = await redis.lrange(CHAT_REDIS_KEY, 0, -1)
        for msg in reversed(recent_msgs):
            await ws.send_text(msg)

    @classmethod
    async def disconnect(cls, ws: WebSocket, uid: str):
        ws = cls.sockets.pop(uid, None)
        try:
            await ws.close()
        except RuntimeError:
            pass

    @classmethod
    async def broadcast(cls, uid: str, username: str, message: str):
        # Redis에 메시지 저장 (최근 MAX_MSG만)
        redis = Redis.get_client()
        await redis.lpush(CHAT_REDIS_KEY, f"{username}: {message}")
        await redis.ltrim(CHAT_REDIS_KEY, 0, MAX_MSG - 1)

        # WebSocket으로 메시지 전송
        for uid, ws in cls.sockets.items():
            try:
                await ws.send_text(f"{username}: {message}")
            except:
                await cls.disconnect(ws, uid)


# -------------------------------
# Kafka Producer/Consumer
# -------------------------------
kafka_producer: AIOKafkaProducer = None
kafka_chat_consumer: AIOKafkaConsumer = None
kafka_log_consumer: AIOKafkaConsumer = None


async def start_kafka(loop):
    global kafka_producer, kafka_chat_consumer, kafka_log_consumer

    kafka_producer = AIOKafkaProducer(bootstrap_servers="localhost:9092", loop=loop)
    await kafka_producer.start()

    kafka_chat_consumer = AIOKafkaConsumer(
        "chat-topic",
        bootstrap_servers="localhost:9092",
        group_id="chat-service",
        auto_offset_reset="latest",
        loop=loop,
    )
    await kafka_chat_consumer.start()
    
    kafka_log_consumer = AIOKafkaConsumer(
        "log-topic",
        bootstrap_servers="localhost:9092",
        group_id="log-service",
        auto_offset_reset="latest",
        loop=loop,
    )
    await kafka_log_consumer.start()

    # 백그라운드 Consumer Task 실행
    asyncio.create_task(kafka_chat_consumer_task())
    asyncio.create_task(kafka_log_consumer_task())

async def send_kafka_chat(uid: str, username: str, message: str):
    data = {"uid": uid, "username": username, "msg": message}
    await kafka_producer.send_and_wait("chat-topic", json.dumps(data).encode())
    
async def send_kafka_log(uid: str, type :str, username: str, message: str):
    data = {"type": type, "uid": uid, "username": username, "msg": message}
    log.info(data) # 추후 지울것 (입출력도 많이하면 서버 성능 저하)
    await kafka_producer.send_and_wait("log-topic", json.dumps(data).encode())

# Comsumer Task
async def kafka_chat_consumer_task():
    buffer = []
    last_flush = time.time()
    mongo_db = MongoDB.get_database()
    
    async for msg in kafka_chat_consumer:
        data = json.loads(msg.value.decode())
        buffer.append({
            "uid": data["uid"],
            "username": data["username"],
            "message": data["msg"],
            "timestamp": datetime.now(timezone.utc)
        })

        await ChatRoomManager.broadcast(
            uid=data["uid"], username=data["username"], message=data["msg"]
        )
        
        # Flush 조건
        if time.time() - last_flush > FLUSH_INTERVAL:
            if buffer:
                await mongo_db["chat_logs"].insert_many(buffer)
                buffer.clear()
            last_flush = time.time()


async def kafka_log_consumer_task():
    buffer = []
    last_flush = time.time()
    mongo_db = MongoDB.get_database()

    async for msg in kafka_log_consumer:
        data = json.loads(msg.value.decode())
        buffer.append({
            "type": data["type"],
            "uid": data["uid"],
            "username": data["username"],
            "message": data["msg"],
            "timestamp": datetime.now(timezone.utc)
        })

        # Flush 조건
        if time.time() - last_flush > FLUSH_INTERVAL:
            if buffer:
                await mongo_db["system_logs"].insert_many(buffer)
                buffer.clear()
            last_flush = time.time()

# -------------------------------
# FastAPI Lifespan
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await Redis.connect()
    MongoDB.connect()

    loop = asyncio.get_running_loop()
    await start_kafka(loop)

    yield

    await Redis.close()
    MongoDB.close()

    await kafka_producer.stop()
    await kafka_chat_consumer.stop()
    await kafka_log_consumer.stop()


app = FastAPI(lifespan=lifespan)

app.include_router(auth_router)
app.include_router(users_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code, content={"status": "error", "message": exc.detail}
    )

@app.middleware("http")
async def add_process_time_header(
    request: Request, call_next
) -> Response:
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    session_id = await get_session_id(request)
    user_info = await get_user_info(session_id)
    
    log_data = {
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "process_time": round(process_time, 3),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if user_info:
        await refresh_session(session_id)
        asyncio.create_task(send_kafka_log(
            uid=user_info.uid,
            type="http",
            username=user_info.username,
            message=json.dumps(log_data)
        ))
    else:
        asyncio.create_task(send_kafka_log(
            uid="",
            type="http",
            username="Guest",
            message=json.dumps(log_data)
        ))
    return response

# -------------------------------
# WebSocket Endpoint
# -------------------------------
@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    session_id = await get_session_id(ws)
    user_info = await get_user_info(session_id)
    if not user_info:
        return

    uid = user_info.uid
    await ChatRoomManager.connect(ws, uid)

    try:
        while True:
            data = await ws.receive_text()
            user_info = await get_user_info(session_id)
            
            await refresh_session(session_id)
            await send_kafka_chat(user_info.uid, user_info.username, data)
            
    except WebSocketDisconnect:
        await ChatRoomManager.disconnect(ws, uid)


# -------------------------------
# 테스트용 HTML 페이지
# -------------------------------
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
            // 현재 접속한 도메인/포트 기반으로 WebSocket 연결
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
