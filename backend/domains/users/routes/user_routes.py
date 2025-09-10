from fastapi import APIRouter, HTTPException

from modules.database import MongoDB
from domains.auth.services.auth_service import hash_password
from domains.users.models.user import User
from domains.users.schemes.register_request import RegisterRequest

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/register")
async def register_user(register_data: RegisterRequest) -> dict[str, str]:
    """
    새로운 사용자를 등록합니다.

    Args:
        register_data (RegisterRequest): 사용자 등록 요청 데이터

    Returns:
        dict[str, str]: 등록 성공 메시지
    """
    existing_user = await MongoDB.get_database().users.find_one(
        {"username": register_data.username}
    )
    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists")

    user = User(**register_data.model_dump())
    user.password = hash_password(user.password)
    await MongoDB.get_database().users.insert_one(user.model_dump())

    return {"message": "Register successful"}
