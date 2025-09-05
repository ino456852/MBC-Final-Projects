import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
from pymongo.collection import Collection
import httpx
import pandas as pd


def insert_with_datareader(
    collection: Collection, value_name: str, name: str, data_source: str
):
    latest_doc = collection.find_one(
        sort=[("date", -1)], projection={"date": 1, "_id": 0}
    )

    if latest_doc:
        # MongoDB에 저장된 마지막 날짜 이후부터 가져오기
        start_date = latest_doc["date"] + timedelta(days=1)
    else:
        # 컬렉션이 없거나 비어있으면 10년 전부터 가져오기
        start_date = datetime(2015, 9, 1)

    start_date = start_date.date()
    if start_date >= datetime.today().date():
        return

    data = web.DataReader(name, data_source, start_date)

    records = [
        {"date": idx.to_pydatetime(), value_name: float(val)}
        for idx, val in data[name].items()
    ]
    if records:
        collection.insert_many(records)


def insert_with_yfinance(collection: Collection, value_name: str, ticker: str):
    latest_doc = collection.find_one(
        sort=[("date", -1)], projection={"date": 1, "_id": 0}
    )

    if latest_doc:
        # MongoDB에 저장된 마지막 날짜 이후부터 가져오기
        start_date = latest_doc["date"] + timedelta(days=1)
    else:
        # 컬렉션이 없거나 비어있으면 10년 전부터 가져오기
        start_date = datetime(2015, 9, 1)

    start_date = start_date.date()
    if start_date >= datetime.today().date():
        return

    data = yf.download(ticker, start=start_date, interval="1d")["Close"]

    records = [
        {"date": idx, value_name: float(val)}
        for idx, val in zip(data[ticker].index, data[ticker].values)
    ]

    if records:
        collection.insert_many(records)


def insert_with_ecos(
    collection: Collection,
    value_name: str,
    api_key: str,
    stat_code: str,
    interval: str,
    code: str,
):
    latest_doc = collection.find_one(
        sort=[("date", -1)], projection={"date": 1, "_id": 0}
    )

    # MongoDB에 저장된 마지막 날짜 이후부터 가져오기
    # 컬렉션이 없거나 비어있으면 10년 전부터 가져오기
    if interval == "D":
        date_format = "%Y%m%d"
        if latest_doc:
            start_date = latest_doc["date"] + timedelta(days=1)
            start_date = start_date.strftime(date_format)
        else:
            start_date = "20150901"
    elif interval == "M":
        date_format = "%Y%m"
        if latest_doc:
            year = latest_doc["date"].year
            month = latest_doc["date"].month + 1
            if month > 12:
                month = 1
                year += 1
            start_date = f"{year}{month:02d}"
        else:
            start_date = "201509"

    end_date = datetime.today().strftime(date_format)
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/5000/{stat_code}/{interval}/{start_date}/{end_date}/{code}"

    response = httpx.get(url)
    data = response.json()

    if not data.get("StatisticSearch"):
        return

    rows = data["StatisticSearch"]["row"]

    data = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]].rename(
        columns={"TIME": "date", "DATA_VALUE": value_name}
    )
    data["date"] = pd.to_datetime(data["date"], format=date_format)
    data[value_name] = data[value_name].astype(float)

    records = data.to_dict("records")
    if records:
        collection.insert_many(records)
