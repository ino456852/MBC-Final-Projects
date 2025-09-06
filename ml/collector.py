from database import MongoDB
from config import config
from data_ingest import (
    insert_with_datareader,
    insert_with_yfinance,
    insert_with_ecos,
)


def save_log(name: str):
    print(f"[{name}] DB에 최신데이터를 저장하고있습니다...")


if __name__ == "__main__":
    MongoDB.connect()
    db = MongoDB.get_database()
    eco_api_key = config.ECOS_API_KEY

    # 단위: 퍼센트 (공휴일, 휴장일은 결측치)
    save_log("dgs10")
    insert_with_datareader(
        db["dgs10"], value_name="dgs10", name="DGS10", data_source="fred"
    )

    # 단위: 지수 (1973.03 [고정환율 -> 변동환율 변환 시점] = 100 기준)
    save_log("dxy")
    insert_with_yfinance(db["dxy"], value_name="dxy", ticker="DX-Y.NYB")

    # 한국 기준금리 (월별), 단위: 연%
    save_log("kr_rate")
    insert_with_ecos(
        db["kr_rate"],
        value_name="kr_rate",
        api_key=eco_api_key,
        stat_code="722Y001",
        interval="M",
        code="0101000",
    )

    # 미국 기준금리 (월별), 단위: 퍼센트
    save_log("us_rate")
    insert_with_datareader(
        db["us_rate"], value_name="us_rate", name="FEDFUNDS", data_source="fred"
    )

    # 단위: 퍼센트 (S&P 500 옵션 가격으로 향후 30일간 예상 변동성 추정한 값)
    save_log("vix")
    insert_with_yfinance(db["vix"], value_name="vix", ticker="^VIX")

    # 단위: USD/배럴
    save_log("wti")
    insert_with_yfinance(db["wti"], value_name="wti", ticker="CL=F")

    save_log("usd")
    insert_with_ecos(
        db["usd"],
        value_name="usd",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000001",
    )

    save_log("gbp")
    insert_with_ecos(
        db["gbp"],
        value_name="gbp",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000012",
    )

    save_log("jpy(100)")
    insert_with_ecos(
        db["jpy(100)"],
        value_name="jpy(100)",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000002",
    )

    save_log("eur")
    insert_with_ecos(
        db["eur"],
        value_name="eur",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000003",
    )

    save_log("cny")
    insert_with_ecos(
        db["cny"],
        value_name="cny",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000053",
    )

    save_log("cny_usd")
    # CNY/USD (15.09.01 ~ 15.12.31)
    insert_with_ecos(
        db["cny_usd"],
        value_name="cny_usd",
        api_key=eco_api_key,
        stat_code="731Y002",
        interval="D",
        code="0000027",
    )

    MongoDB.close()
