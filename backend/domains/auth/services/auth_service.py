from datetime import datetime, timedelta, timezone
from uuid import uuid4
from passlib.context import CryptContext

from core.database import MongoDB

SESSION_TTL = 3600  # 1 hour
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# 세션 생성
async def create_session(user_id: str) -> str:
    session_id = str(uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL)
    await MongoDB.get_database().sessions.insert_one({
        "_id": session_id,
        "user_id": user_id,
        "expires_at": expires_at
    })
    return session_id