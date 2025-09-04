# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

class Config:
    MONGO_URI = os.getenv("MONGO_URI")  # 이미 ID/PW 포함된 URI라고 가정
    MONGO_DB = os.getenv("MONGO_DB") # DB 이름
    DEBUG = os.getenv("DEBUG", "False").lower() in ('true', '1', 't')

config = Config()