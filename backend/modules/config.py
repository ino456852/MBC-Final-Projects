# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    REDIS_URI = os.getenv("REDIS_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")


config = Config()
