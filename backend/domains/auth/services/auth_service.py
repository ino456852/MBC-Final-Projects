from datetime import datetime, timedelta, timezone
import re
from typing import Dict, Optional, Union
from uuid import uuid4
from fastapi import HTTPException, Request, WebSocket
from passlib.context import CryptContext
from domains.users.schemes.user_info import UserInfo
from core.database import MongoDB

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SESSION_TTL = 3600  # 1 hour
SESSION_ID_PATTERN = re.compile(r"session_id=([^;]+)")

def hash_password(password: str) -> str:
    """
    비밀번호를 해시합니다.

    Args:
        password (str): 평문 비밀번호

    Returns:
        str: 해시된 비밀번호
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    비밀번호 일치 여부를 확인합니다.

    Args:
        plain_password (str): 평문 비밀번호
        hashed_password (str): 해시된 비밀번호

    Returns:
        bool: 일치 여부
    """
    return pwd_context.verify(plain_password, hashed_password)


async def create_session(user_id: str) -> str:
    """
    사용자 세션을 생성하고 MongoDB에 저장합니다.

    Args:
        user_id (str): 사용자 ID

    Returns:
        str: 세션 ID
    """
    session_id = str(uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL)
    await MongoDB.get_database().sessions.insert_one({
        "_id": session_id,
        "user_id": user_id,
        "expires_at": expires_at
    })
    return session_id


async def get_user_info(session_id: str | None) -> UserInfo:
    """
    현재 로그인한 사용자의 정보를 반환합니다.

    Args:
        session_id (str | None): 클라이언트 쿠키에 저장된 세션 ID

    Returns:
        UserInfo: 현재 로그인한 사용자의 정보
    """
    if not session_id:
        return None

    session = await MongoDB.get_database().sessions.find_one({
        "_id": session_id,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })

    if not session:
        return None

    user = await MongoDB.get_database().users.find_one({"_id": session["user_id"]})

    if not user:
        return None

    return UserInfo(**user)

async def get_session_id(conn: Union[WebSocket, Request]) -> str:
    """
    WebSocket 또는 HTTP Request에서 쿠키를 읽어
    채팅 가능한 사용자인지 검증하고 session_id와 사용자 정보 반환.
    인증 실패 시 None 반환.
    """
    try:
        headers = conn.headers  # WebSocket도 request와 동일하게 headers 속성 있음
        cookie_header = headers.get("cookie", "")
        match = SESSION_ID_PATTERN.search(cookie_header)
        
        if not match:
            return None

        session_id = match.group(1)

        user_info = get_user_info(session_id)
        if not user_info:
            return None

        return session_id

    except Exception:
        return None