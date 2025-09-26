from fastapi import APIRouter
from modules.mongodb import MongoDB

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/")
async def register_user():
    db = MongoDB.get_database()
    return {"message": "hihih"}
