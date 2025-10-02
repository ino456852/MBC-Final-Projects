from fastapi import APIRouter, Query
from modules.mongodb import MongoDB
import pandas as pd
from ml.data_merge import create_merged_dataset

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/")
async def dashboard(
    currency: str = Query("usd", description="조회할 통화 코드 (예: usd, eur, jpy)")
):
    db = MongoDB.get_database()

    # CSV 읽기 (날짜를 인덱스로)
    merged_data = create_merged_dataset()
    merged_data = merged_data[[currency]]  # 원하는 통화 열만 선택
    merged_data.index = merged_data.index.strftime("%Y-%m-%d")

    # 이동평균 계산
    windows = [5, 20, 60, 120]
    for w in windows:
        merged_data[f"SMA_{w}"] = merged_data[currency].rolling(window=w).mean()
        merged_data[f"EMA_{w}"] = merged_data[currency].ewm(span=w, adjust=False).mean()

    # NaN 제거
    merged_data = merged_data.dropna()

    # real_prices 변환
    real_prices = [
        {"date": idx, **{col: row[col] for col in merged_data.columns}}
        for idx, row in merged_data.iterrows()
    ]

    # MongoDB 조회
    cursor = db["predicted_price"].find(
        {}, {"_id": 0, "model": 1, "date": 1, currency: 1}
    )
    results = await cursor.to_list(length=None)
    df = pd.DataFrame(results)

    # predicted_prices 변환
    predicted_df = df.pivot(index="date", columns="model", values=currency).reset_index()
    predicted_prices = predicted_df.to_dict(orient="records")

    return {
        "real_prices": real_prices,
        "predicted_prices": predicted_prices
    }
