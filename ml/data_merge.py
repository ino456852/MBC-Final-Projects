from datetime import datetime

import pandas as pd
from database import MongoDB
from data_import import import_from_db


def create_merged_dataset():
    MongoDB.connect()
    db = MongoDB.get_database()

    try:
        # 1. 종속변수 (환율) 가져오기 및 병합
        usd = import_from_db(db["usd"])
        gbp = import_from_db(db["gbp"])
        eur = import_from_db(db["eur"])
        jpy = import_from_db(db["jpy(100)"]).rename(columns={"jpy(100)": "jpy"})

        # 과거엔 usd_cny만 있어서 과거데이터는 추가 계산이 필요
        # 계산후 최신 데이터와 병합
        usd_cny = import_from_db(db["usd_cny"])
        usd_cny["cny"] = usd["usd"] / usd_cny["usd_cny"]
        usd_cny = usd_cny[["date", "cny"]]
        cny = import_from_db(db["cny"])
        cny_merged = pd.concat([usd_cny, cny], axis=0).reset_index(drop=True)

        # 2. 독립변수 데이터 가져오기 및 병합
        vix = import_from_db(db["vix"])
        dxy = import_from_db(db["dxy"])
        wti = import_from_db(db["wti"])
        dgs10 = import_from_db(db["dgs10"])
        kr_rate = import_from_db(db["kr_rate"])
        us_rate = import_from_db(db["us_rate"])

        data_list = [
            usd,
            gbp,
            eur,
            jpy,
            cny_merged,
            vix,
            dxy,
            wti,
            dgs10,
            kr_rate,
            us_rate,
        ]

        for i, df in enumerate(data_list):
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # 누락 날짜 추가 후 ffill
            today = pd.to_datetime(datetime.today().date())
            all_dates = pd.date_range("2015-09-01", today)

            # 변경된 dataframe 저장
            data_list[i] = df.reindex(all_dates).ffill()

        df = pd.concat(data_list, axis=1, join="outer")
        return df

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None

    finally:
        MongoDB.close()


if __name__ == "__main__":
    dataset = create_merged_dataset().round(4)
    if dataset is not None:
        dataset.to_csv("dataset.csv", encoding="utf-8-sig")
