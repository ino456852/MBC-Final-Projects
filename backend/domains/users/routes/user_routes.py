from datetime import datetime, timezone
from fastapi import APIRouter, Cookie, HTTPException

from core.database import MongoDB
from domains.auth.services.auth_service import hash_password
from domains.users.models.user import User
from domains.users.schemes.register_request import RegisterRequest
from domains.users.schemes.user_info import UserInfo

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
async def current_user(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="No session provided")
    
    session = await MongoDB.get_database().sessions.find_one({
        "_id": session_id,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    user = await MongoDB.get_database().users.find_one({
        "_id": session["user_id"]
    })
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return UserInfo(**user)


@router.get("/point-ranks")
async def point_ranks():
    cursor = MongoDB.get_database().users.find().sort("point", -1).limit(10)
    
    user_list = await cursor.to_list(length=10)
    top10_users = [UserInfo(**user) for user in user_list]
    
    return top10_users


@router.post("/register")
async def register(register_data: RegisterRequest):
    user = await MongoDB.get_database().users.find_one({"username": register_data.username})
    
    if user:
        raise HTTPException(status_code=409, detail="User already exists")
    
    # 비밀번호 해싱
    user = User(**register_data.model_dump())
    user.password = hash_password(user.password)
    await MongoDB.get_database().users.insert_one(user.model_dump())
    
    return {"message": "User registered successfully"}