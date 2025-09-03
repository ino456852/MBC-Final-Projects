# modules/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGO_URI = os.getenv("MONGO_URI")
    DEBUG = os.getenv("DEBUG", "False").lower() in ('true', '1', 't')
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-change-me")

settings = Settings()