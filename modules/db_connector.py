# modules/db_connector.py
from pymongo import MongoClient
from .config import settings
from .logger import log

def get_database():
    try:
        client = MongoClient(settings.MONGO_URI)
        # ping을 통해 연결 유효성 검사
        client.admin.command('ping')
        log.info("✅ MongoDB에 성공적으로 연결되었습니다.")
        
        # 사용할 데이터베이스를 선택합니다.
        db = client['mbc_final_project_db'] # 실제 사용할 DB 이름으로 변경
        return db
    except Exception as e:
        log.critical(f"❌ MongoDB 연결에 실패했습니다: {e}", exc_info=True)
        return None