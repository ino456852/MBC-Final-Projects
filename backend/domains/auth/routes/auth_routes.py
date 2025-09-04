from fastapi import APIRouter, Cookie, HTTPException, Response
from core.database import MongoDB
from domains.auth.shcemes.login_request import LoginRequest
from domains.auth.services.auth_service import SESSION_TTL, create_session, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
async def login(login_data: LoginRequest, response: Response):
    
    user = await MongoDB.get_database().users.find_one({"username": login_data.username})
    if not user or not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session_id = await create_session(user["_id"])

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=SESSION_TTL
    )

    return {"message": "Login successful"}

# 로그아웃
@router.post("/logout")
async def logout(response: Response, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="No session provided")

    result = await MongoDB.get_database().sessions.delete_one({"_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    response.delete_cookie(
        key="session_id",
        httponly=True,
        secure=True,
        samesite="lax"
    )

    return {"message": "Logout successful"}
