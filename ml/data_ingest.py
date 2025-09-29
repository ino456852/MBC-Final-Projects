import pandas_datareader.data as web
import pandas as pd
import yfinance as yf
import httpx
import requests
from datetime import datetime, timedelta
from pymongo.database import Database
from bs4 import BeautifulSoup
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

        # 기존 컬렉션의 마지막 날짜 조회
        latest_doc = collection.find_one(sort=[("date", -1)], projection={"date": 1, "_id": 0})
        existing_dates = set(collection.distinct("date"))

        # 시작일 계산
        if latest_doc:
            start_date = latest_doc["date"]
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            # interval에 맞춰 다음 날짜부터 시작
            start_date = next_date(start_date, interval)
        else:
            # 컬렉션 비어있으면 기본값
            start_date = "201509" if interval == "M" else "20150901"

        # 문자열 → datetime 변환
        if interval == "M":
            start_date = datetime.strptime(start_date, "%Y%m").date()
        else:
            start_date = datetime.strptime(start_date, "%Y%m%d").date()

        if start_date >= datetime.today().date():
            return

        # 데이터 가져오기
        data = web.DataReader(reader_name, data_source, start_date)

        records = []
        for idx, val in data[reader_name].items():
            dt = pd.to_datetime(idx).date()  # 문자열/타임스탬프 모두 처리
            if dt not in existing_dates:
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

        latest_doc = collection.find_one(
            sort=[("date", -1)], projection={"date": 1, "_id": 0}
        )

        if latest_doc:
            start_date = next_date(latest_doc["date"], "D")
        else:
            start_date = "2015-09-01"

        start_date = datetime.strptime(start_date, "%Y%m%d").date()
        if start_date >= datetime.today().date():
            return

        data = yf.download(ticker, start=start_date, interval="1d")["Close"]

        if isinstance(data, pd.Series):
            records = [
                {"date": pd.to_datetime(idx), coll_name: float(val)}
                for idx, val in data.items()
                if not pd.isna(val)
            ]
        else:  # DataFrame
            records = [
                {"date": pd.to_datetime(idx), coll_name: float(val)}
                for idx, val in zip(data.index, data[ticker].values)
                if not pd.isna(val)
            ]

        if not records:
            return
        
        existing_dates = {doc["date"] for doc in collection.find({}, {"date": 1})}
        records = [r for r in records if r["date"] not in existing_dates]

        if not records:
            return

        result = collection.insert_many(records)
        count = len(result.inserted_ids)
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)

# 한국은행 ECOS 데이터 MongoDB 컬렉션에 삽입
def insert_with_ecos(
    db: Database, coll_name: str, api_key: str, stat_code: str, interval: str, code: str,
    start_date: Optional[str] = None, end_date: Optional[str] = None):

    count = 0
    try:
        # interval별 설정
        date_format, default_start = {
            "D": ("%Y%m%d", "20150901"),
            "M": ("%Y%m", "201509"),
            "Y": ("%Y", "2015"),
        }[interval]

        collection = db[coll_name]

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

# Investing.com 데이터를 MongoDB 컬렉션에 삽입
def insert_with_investing(db: Database, coll_name: str):
    count = 0
    try:
        collection = db[coll_name]
        
        latest_doc = collection.find_one(sort=[("date", -1)], projection={"date": 1, "_id": 0})
        if latest_doc:
            start_date = latest_doc["date"] + timedelta(days=1)
        else:
            start_date = datetime(2015, 9, 1)

        if start_date >= datetime.today():
            return

        url = "https://www.investing.com/rates-bonds/china-10-year-bond-yield-historical-data"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")

        table = soup.find("table", {"id": "curr_table"})
        df = pd.read_html(StringIO(str(table)))[0]

        df = df.rename(columns={"Date": "date", "Price": coll_name})
        df[coll_name] = df[coll_name].astype(str).str.replace("%", "").astype(float)
        df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")

        df = df[df["date"] >= start_date]
        
        existing_dates = set(collection.distinct("date"))
        df = df[~df["date"].isin(existing_dates)]

        if df.empty:
            return

        records = [{"date": row["date"], coll_name: row[coll_name]} for _, row in df.iterrows()]
        result = collection.insert_many(records)
        count = len(result.inserted_ids)

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        insert_log(coll_name, count)