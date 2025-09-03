import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

df = pd.read_csv("data/주요국 통화의 대원화환율_02150728.csv")

print(df.head(5))

print(df['CNY'].isnull().sum())

# 1. 'Date'를 기준으로 데이터를 반드시 먼저 정렬합니다.
df = df.sort_values(by='Date')

print("--- 결측치 처리 전 ---")
# 위안화(CNY) 데이터의 맨 앞부분을 확인하면 NaN이 많을 것입니다.
print(df.head())
print("\CNY 컬럼의 결측치 개수:", df['CNY'].isnull().sum())


# 2. *** 여기가 핵심 해결책! ***
# 1단계: ffill로 중간 결측치 채우기 (앞사람 보고 베끼기)
df_filled = df.fillna(method='ffill')

# 2단계: bfill로 맨 앞 결측치 채우기 (뒷사람 보고 베끼기)
df_filled = df_filled.fillna(method='bfill')


print("\n--- 결측치 처리 후 ---")
# 이제 위안화(CNY) 데이터의 맨 앞부분이 채워졌을 것입니다.
print(df_filled.head())
print("\CNY 컬럼의 결측치 개수:", df_filled['CNY'].isnull().sum())

# print(df_filled.dtype)
df_filled.to_csv("exchange_rates_fix.csv",index=False)

# =================================================================================
# 3. MongoDB Atlas에 업로드하기 (이 부분을 추가!)
# =================================================================================

# --- 3.1 연결 정보 설정 ---
# Atlas에서 복사한 '비밀 연결 주소'를 여기에 붙여넣으세요.
# <password> 부분은 실제 비밀번호로 반드시 교체해야 합니다!
CONNECTION_STRING = "mongodb+srv://mbc_final_user:UqAM7Z2eZ6ZQ11Ra@mbc-final-cluster.g9pivwk.mongodb.net/?retryWrites=true&w=majority&appName=mbc-final-cluster"

DB_NAME = "exchange_rate_db"
COLLECTION_NAME = "daily_exchange_rates" # 컬렉션 이름

# --- 3.2 데이터베이스 연결 및 업로드 ---
try:
    # 1. MongoDB 클라이언트를 생성하여 서버에 연결합니다.
    client = MongoClient(CONNECTION_STRING)
    
    # 2. 연결이 성공했는지 확인합니다 (선택 사항이지만 좋은 습관).
    client.admin.command('ping')
    print("\nMongoDB Atlas에 성공적으로 연결되었습니다!")
    
    # 3. 사용할 데이터베이스를 선택합니다 (없으면 자동으로 생성됩니다).
    db = client[DB_NAME]
    
    # 4. 사용할 컬렉션을 선택합니다 (없으면 자동으로 생성됩니다).
    collection = db[COLLECTION_NAME]
    
    # 5. (선택 사항) 기존 데이터를 모두 지우고 새로 시작하고 싶을 때 이 줄의 주석을 푸세요.
    # collection.delete_many({})
    # print(f"'{COLLECTION_NAME}' 컬렉션의 기존 데이터가 모두 삭제되었습니다.")
    
    # 6. pandas DataFrame을 MongoDB가 이해할 수 있는 dictionary 리스트로 변환합니다.
    # 'Date' 컬럼을 문자열로 변환해야 JSON 직렬화 오류를 피할 수 있습니다.
    df_filled['Date'] = df_filled['Date'].astype(str)
    data_to_upload = df_filled.to_dict("records")
    
    # 7. 변환된 데이터를 컬렉션에 한 번에 삽입합니다.
    if data_to_upload: # 업로드할 데이터가 있을 경우에만 실행
        collection.insert_many(data_to_upload)
        print(f"\n총 {len(data_to_upload)}개의 데이터가 '{DB_NAME}.{COLLECTION_NAME}'에 성공적으로 업로드되었습니다.")
    else:
        print("\n업로드할 데이터가 없습니다.")

except ConnectionFailure as e:
    print(f"MongoDB 연결 실패: {e}")
except Exception as e:
    print(f"데이터 업로드 중 오류가 발생했습니다: {e}")
finally:
    # 8. 모든 작업이 끝나면 연결을 닫아주는 것이 좋습니다.
    if 'client' in locals() and client:
        client.close()
        print("MongoDB 연결이 닫혔습니다.")