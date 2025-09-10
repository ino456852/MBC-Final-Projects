from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from .config import config
from .logger import log
import redis.asyncio as aioredis


class Redis:
    _client: aioredis.Redis | None = None

    @classmethod
    async def connect(cls):
        """Redis 클라이언트 초기화"""
        if cls._client is None:
            cls._client = aioredis.from_url(
                config.REDIS_URI,
                decode_responses=True,  # 문자열 반환을 위해
            )
            # 연결 테스트
            try:
                await cls._client.ping()
                log.info("✅ Redis connected")
            except Exception as e:
                log.error(f"❌ Redis connection failed: {e}")
                cls._client = None
                raise

    @classmethod
    async def close(cls):
        """Redis 연결 종료"""
        if cls._client:
            await cls._client.close()
            cls._client = None
            log.info("🛑 Redis connection closed")

    @classmethod
    def get_client(cls) -> aioredis.Redis:
        """Redis 클라이언트 가져오기"""
        if cls._client is None:
            raise RuntimeError(
                "❌ Redis not connected. Did you forget to call Redis.connect()?"
            )
        return cls._client


class MongoDB:
    _client: AsyncIOMotorClient | None = None
    _db: AsyncIOMotorDatabase | None = None

    @classmethod
    def connect(cls):
        """MongoDB 클라이언트 초기화"""
        if cls._client is None:
            cls._client = AsyncIOMotorClient(config.MONGO_URI)
            cls._db = cls._client[config.MONGO_DB_NAME]
            log.info("✅ MongoDB connected")

    @classmethod
    def close(cls):
        """MongoDB 연결 종료"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            log.info("🛑 MongoDB connection closed")

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """DB 핸들 가져오기"""
        if cls._db is None:
            raise RuntimeError(
                "❌ MongoDB not connected. Did you forget to call MongoDB.connect()?"
            )
        return cls._db
