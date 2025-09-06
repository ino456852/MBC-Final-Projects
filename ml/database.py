from pymongo import MongoClient
from pymongo.database import Database
from config import config


class MongoDB:
    """
    MongoDB 연결을 관리하는 유틸리티 클래스
    """

    _client: MongoClient | None = None
    _db: Database | None = None

    @classmethod
    def connect(cls):
        """
        MongoDB 클라이언트를 초기화합니다.
        """
        if cls._client is None:
            cls._client = MongoClient(config.MONGO_URI)
            cls._db = cls._client[config.MONGO_DB_NAME]
            print("✅ MongoDB connected")

    @classmethod
    def close(cls):
        """
        MongoDB 연결을 종료합니다.
        """
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("🛑 MongoDB connection closed")

    @classmethod
    def get_database(cls) -> Database:
        """
        데이터베이스 핸들을 반환합니다.

        return:
            Database: pymongo Database 객체
        """
        if cls._db is None:
            raise RuntimeError(
                "❌ MongoDB not connected. Did you call MongoDB.connect()?"
            )
        return cls._db


if __name__ == "__main__":
    print("MONGO DB 테스트:")
    MongoDB.connect()
    db = MongoDB.get_database()
    print("컬렉션 목록:", db.list_collection_names())
    MongoDB.close()
