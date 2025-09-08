from typing import Optional
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
import httpx
import pandas as pd
from pymongo.database import Database


def insert_log(name: str, count: int):
    print(f"[{name}] 컬렉션에 {count}개를 저장했습니다")


def next_date(date: datetime, interval: str) -> str:
    if interval == "D":
        return (date + timedelta(days=1)).strftime("%Y%m%d")
    if interval == "M":
        year, month = date.year, date.month + 1
        if month > 12:
            year, month = year + 1, 1
        return f"{year}{month:02d}"
    if interval == "Y":
        return str(date.year + 1)
    raise ValueError(f"Unsupported interval: {interval}")


def insert_with_datareader(
    db: Database, coll_name: str, reader_name: str, data_source: str
):
    """Pandas Datareader API 데이터를 MongoDB 컬렉션에 삽입"""

    count = 0
    try:
        collection = db[coll_name]

        latest_doc = collection.find_one(
            sort=[("date", -1)], projection={"date": 1, "_id": 0}
        )

        if latest_doc:
            # MongoDB에 저장된 마지막 날짜 이후부터 가져오기
            start_date = next_date(latest_doc["date"], "D")
        else:
            # 컬렉션이 없거나 비어있으면 10년 전부터 가져오기
            start_date = "20150901"

        start_date = datetime.strptime(start_date, "%Y%m%d").date()
        if start_date >= datetime.today().date():
            return

        data = web.DataReader(reader_name, data_source, start_date)

        records = [
            {"date": idx.to_pydatetime(), coll_name: float(val)}
            for idx, val in data[reader_name].items()
        ]

        if not records:
            return

        result = collection.insert_many(records)
        count = len(result.inserted_ids)
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)


def insert_with_yfinance(db: Database, coll_name: str, ticker: str):
    """YFinance API 데이터를 MongoDB 컬렉션에 삽입"""

    count = 0

    try:
        collection = db[coll_name]

        latest_doc = collection.find_one(
            sort=[("date", -1)], projection={"date": 1, "_id": 0}
        )

        if latest_doc:
            # MongoDB에 저장된 마지막 날짜 이후부터 가져오기
            start_date = next_date(latest_doc["date"], "D")
        else:
            # 컬렉션이 없거나 비어있으면 10년 전부터 가져오기
            start_date = "2015-09-01"

        start_date = datetime.strptime(start_date, "%Y%m%d").date()
        if start_date >= datetime.today().date():
            return

        data = yf.download(ticker, start=start_date, interval="1d")["Close"]

        records = [
            {"date": idx, coll_name: float(val)}
            for idx, val in zip(data[ticker].index, data[ticker].values)
        ]

        if not records:
            return

        result = collection.insert_many(records)
        count = len(result.inserted_ids)
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)


def insert_with_ecos(
    db: Database,
    coll_name: str,
    api_key: str,
    stat_code: str,
    interval: str,
    code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """한국은행 ECOS API 데이터를 MongoDB 컬렉션에 삽입"""

    count = 0

    try:
        # interval별 설정
        date_format, default_start = {
            "D": ("%Y%m%d", "20150901"),
            "M": ("%Y%m", "201509"),
            "Y": ("%Y", "2015"),
        }[interval]

        collection = db[coll_name]

        # MongoDB에서 마지막 날짜 조회
        latest_doc = collection.find_one(
            sort=[("date", -1)], projection={"date": 1, "_id": 0}
        )

        oldest_doc = collection.find_one(
            sort=[("date", 1)], projection={"date": 1, "_id": 0}
        )

        if latest_doc:
            next_dt = next_date(latest_doc["date"], interval)
            if start_date:
                # start_date와 DB 다음 날짜 중 큰 값을 선택
                start_date = max(next_dt, start_date)
            else:
                start_date = next_dt
        else:
            # DB가 비어있으면 기본값 사용
            if not start_date:
                start_date = default_start

        # 종료일 없으면 오늘 날짜
        end_date = end_date or datetime.today().strftime(date_format)

        start_dt = datetime.strptime(start_date, date_format)
        end_dt = datetime.strptime(end_date, date_format)

        if oldest_doc and latest_doc:
            oldest_dt = oldest_doc["date"]
            latest_dt = latest_doc["date"]
            if oldest_dt <= start_dt <= latest_dt or oldest_dt <= end_dt <= latest_dt:
                return

        if start_dt > end_dt:
            return

        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/5000/{stat_code}/{interval}/{start_date}/{end_date}/{code}"

        response = httpx.get(url)
        data = response.json()

        if not data.get("StatisticSearch"):
            return

        rows = data["StatisticSearch"]["row"]

        data = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]].rename(
            columns={"TIME": "date", "DATA_VALUE": coll_name}
        )
        data["date"] = pd.to_datetime(data["date"], format=date_format)
        data[coll_name] = data[coll_name].astype(float)

        records = data.to_dict("records")
        if not records:
            return

        result = collection.insert_many(records)
        count = len(result.inserted_ids)
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)
