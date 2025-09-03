from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# 1. 만들어둔 공통 모듈들을 모두 임포트합니다.
from modules.db_connector import get_database
from modules.logger import log
from modules.response_builder import success_response, error_response
from modules.config import settings

# 2. FastAPI 앱을 생성합니다.
app = FastAPI()

# 3. 앱 시작 시 데이터베이스에 연결합니다.
db = get_database()

# 4. Pydantic으로 요청 Body의 데이터 모델을 정의합니다.
# 이 모델 덕분에 FastAPI가 자동으로 데이터 유효성 검사를 해줍니다.
class Item(BaseModel):
    name: str = Field(..., min_length=2, description="상품 이름")
    price: int = Field(..., gt=0, description="상품 가격은 0보다 커야 합니다.")
    description: Optional[str] = Field(None, description="상품 설명")

# 5. API 엔드포인트를 만듭니다.
@app.get("/")
def health_check():
    """서버 상태를 확인하는 기본 엔드포인트"""
    log.info("Health check endpoint called.")
    return success_response(message=f"Server is running! (Debug Mode: {settings.DEBUG})")

@app.post("/items")
def create_item(item: Item):
    """새로운 상품을 데이터베이스에 추가하는 엔드포인트"""
    # DB 연결이 실패했을 경우에 대한 방어 코드
    if db is None:
        log.error("데이터베이스 연결이 유실되어 상품을 생성할 수 없습니다.")
        return error_response("서버 내부 오류: DB 연결 실패", 500)
    
    try:
        log.info(f"새 상품 추가 시도: {item.name}")
        item_dict = item.dict()

        # 이름이 중복된 상품이 있는지 확인
        if db.items.find_one({"name": item.name}):
            log.warning(f"이미 존재하는 상품명입니다: {item.name}")
            return error_response("이미 존재하는 상품명입니다.", 409) # 409 Conflict

        result = db.items.insert_one(item_dict)
        log.info(f"상품 추가 완료: {item.name}, ID: {result.inserted_id}")
        
        return success_response(
            data={"item_id": str(result.inserted_id)},
            message="상품이 성공적으로 추가되었습니다.",
            status_code=201 # 201 Created
        )
    except Exception as e:
        log.error(f"상품 추가 중 오류 발생: {e}", exc_info=True)
        return error_response("상품 추가 중 서버 오류가 발생했습니다.", 500)