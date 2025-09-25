# -------------------------------
# ChatRoomManager
# -------------------------------
from typing import Dict
from fastapi import WebSocket
from modules.redis import Redis

MAX_MSG = 100  # Redis 캐시 최대 메시지 수
CHAT_REDIS_KEY = "chat"

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