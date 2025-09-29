import pandas_datareader.data as web
import pandas as pd
import yfinance as yf
import httpx
from datetime import datetime, timedelta
from pymongo.database import Database
from typing import Optional
from io import StringIO

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

# FRED 데이터 MongoDB 컬렉션에 삽입
def insert_with_datareader(db: Database, coll_name: str, reader_name: str, data_source: str, interval: str = "D"):
    count = 0
    try:
        collection = db[coll_name]

        latest_doc = collection.find_one(sort=[("date", -1)], projection={"date": 1, "_id": 0})
        start_date = next_date(latest_doc["date"], interval) if latest_doc else "20150901"
        
        if interval == "M":
            start_date = datetime.strptime(start_date, "%Y%m").date()
        else:
            start_date = datetime.strptime(start_date, "%Y%m%d").date()

        if start_date >= datetime.today().date():
            return

        data = web.DataReader(reader_name, data_source, start_date)

        existing_dates = set(collection.distinct("date"))

        records = []
        for idx, val in data[reader_name].items():
            dt = pd.to_datetime(idx).date()
            if dt not in existing_dates and not pd.isna(val):
                records.append({"date": dt, coll_name: float(val)})

        if not records:
            return

        result = collection.insert_many(records)
        count = len(result.inserted_ids)

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)

# YFinance 데이터 MongoDB 컬렉션에 삽입
def insert_with_yfinance(db: Database, coll_name: str, ticker: str):
    count = 0
    try:
        collection = db[coll_name]

        # 기존 컬렉션에서 마지막 날짜 조회
        latest_doc = collection.find_one(sort=[("date", -1)], projection={"date": 1, "_id": 0})
        start_date = next_date(latest_doc["date"], "D") if latest_doc else "2015-09-01"
        start_date = datetime.strptime(start_date, "%Y%m%d").date()
        if start_date >= datetime.today().date():
            return

        # YFinance에서 데이터 가져오기
        data = yf.download(ticker, start=start_date, interval="1d")["Close"]

        # 기존 컬렉션에 있는 모든 날짜 조회 (중복 방지용)
        existing_dates = set(collection.distinct("date"))

        # records 생성, 중복 날짜 제거
        records = []
        if isinstance(data, pd.Series):
            for idx, val in data.items():
                dt = pd.to_datetime(idx)
                if dt not in existing_dates and not pd.isna(val):
                    records.append({"date": dt, coll_name: float(val)})
        else:  # DataFrame
            for idx, val in zip(data.index, data[ticker].values):
                dt = pd.to_datetime(idx)
                if dt not in existing_dates and not pd.isna(val):
                    records.append({"date": dt, coll_name: float(val)})

        if not records:
            return

        # MongoDB에 삽입
        result = collection.insert_many(records)
        count = len(result.inserted_ids)

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)

# 한국은행 ECOS 데이터 MongoDB 컬렉션에 삽입
def insert_with_ecos(
    db: Database, coll_name: str, api_key: str, stat_code: str, interval: str, code: str,
    start_date: Optional[str] = None, end_date: Optional[str] = None
):
    count = 0
    try:
        date_format, default_start = {
            "D": ("%Y%m%d", "20150901"),
            "M": ("%Y%m", "201509"),
            "Y": ("%Y", "2015"),
        }[interval]

        collection = db[coll_name]

        latest_doc = collection.find_one(sort=[("date", -1)], projection={"date": 1, "_id": 0})
        oldest_doc = collection.find_one(sort=[("date", 1)], projection={"date": 1, "_id": 0})

        if latest_doc:
            next_dt = next_date(latest_doc["date"], interval)
            start_date = max(start_date, next_dt) if start_date else next_dt
        else:
            if not start_date:
                start_date = default_start

        end_date = end_date or datetime.today().strftime(date_format)
        start_dt = datetime.strptime(start_date, date_format)
        end_dt = datetime.strptime(end_date, date_format)

        if start_dt > end_dt:
            return

        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/5000/{stat_code}/{interval}/{start_date}/{end_date}/{code}"
        response = httpx.get(url)
        data_json = response.json()

        if not data_json.get("StatisticSearch"):
            return

        rows = data_json["StatisticSearch"]["row"]
        if not rows:
            return

        data = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]].rename(
            columns={"TIME": "date", "DATA_VALUE": coll_name}
        )

        data["date"] = pd.to_datetime(data["date"], format=date_format, errors="coerce")
        data = data.dropna(subset=["date"])  # 잘못된 날짜 제거
        data[coll_name] = pd.to_numeric(data[coll_name], errors="coerce")
        data = data.dropna(subset=[coll_name])  # 잘못된 값 제거

        existing_dates = set(collection.distinct("date"))
        data = data[~data["date"].isin(existing_dates)]
        if data.empty:
            return

        records = data.to_dict("records")
        result = collection.insert_many(records)
        count = len(result.inserted_ids)

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)