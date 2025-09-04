from pydantic import BaseModel

class UserInfo(BaseModel):
    username: str
    point: int