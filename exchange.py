import datetime
import pandas as pd
import requests

# 환율
from datetime import datetime

API_KEY = "JLIMTOGUHFYU4IMR7A3T"
stat_code = "731Y001"
start_date = "20150901"
end_date = datetime.today().strftime("%Y%m%d")
usd = "0000001"
gbp = "0000012"
jpy = "0000002"
eur = "0000003"
cny = "0000053"

# USD
url_usd = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/5000/{stat_code}/D/{start_date}/{end_date}/{usd}/"
response = requests.get(url_usd)
data = response.json()
rows = data['StatisticSearch']['row']
df_usd = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'USD'})
df_usd['DATE'] = pd.to_datetime(df_usd['DATE'], format='%Y%m%d')
df_usd['USD'] = df_usd['USD'].astype(float)

# GBP
url_gbp = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/5000/{stat_code}/D/{start_date}/{end_date}/{gbp}/"
response = requests.get(url_gbp)
data = response.json()
rows = data['StatisticSearch']['row']
df_gbp = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'GBP'})
df_gbp['DATE'] = pd.to_datetime(df_gbp['DATE'], format='%Y%m%d')
df_gbp['GBP'] = df_gbp['GBP'].astype(float)

# JPY(100)
url_jpy = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/5000/{stat_code}/D/{start_date}/{end_date}/{jpy}/"
response = requests.get(url_jpy)
data = response.json()
rows = data['StatisticSearch']['row']
df_jpy = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'JPY(100)'})
df_jpy['DATE'] = pd.to_datetime(df_jpy['DATE'], format='%Y%m%d')
df_jpy['JPY(100)'] = df_jpy['JPY(100)'].astype(float)

# EUR
url_eur = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/5000/{stat_code}/D/{start_date}/{end_date}/{eur}/"
response = requests.get(url_eur)
data = response.json()
rows = data['StatisticSearch']['row']
df_eur = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'EUR'})
df_eur['DATE'] = pd.to_datetime(df_eur['DATE'], format='%Y%m%d')
df_eur['EUR'] = df_eur['EUR'].astype(float)

# CNY
url_cny = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/5000/{stat_code}/D/{start_date}/{end_date}/{cny}/"
response = requests.get(url_cny)
data = response.json()
rows = data['StatisticSearch']['row']
df_cny = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'CNY'})
df_cny['DATE'] = pd.to_datetime(df_cny['DATE'], format='%Y%m%d')
df_cny['CNY'] = df_cny['CNY'].astype(float)

# CNY/USD (15.09.01 ~ 15.12.31): 한화 변경에 필요한 데이터로 학습에는 필요 없는 변수
API_KEY = "JLIMTOGUHFYU4IMR7A3T"
stat_code = "731Y002"
start_date = "20150901"
end_date = "20151231"
cny_usd = "0000027"

url_cny_usd = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/5000/{stat_code}/D/{start_date}/{end_date}/{cny_usd}/"
response = requests.get(url_cny_usd)
data = response.json()
rows = data['StatisticSearch']['row']
df_cny_usd = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'CNY_USD'})
df_cny_usd['DATE'] = pd.to_datetime(df_cny_usd['DATE'], format='%Y%m%d')
df_cny_usd['CNY_USD'] = df_cny_usd['CNY_USD'].astype(float)

# 파일 병합
df_merged = df_usd.merge(df_jpy, on='DATE', how='outer') \
                  .merge(df_eur, on='DATE', how='outer') \
                  .merge(df_gbp, on='DATE', how='outer') \
                  .merge(df_cny, on='DATE', how='outer') \
                  .merge(df_cny_usd, on='DATE', how='outer')

df_merged.head()

# CNY/KRW 변환
mask = (df_merged['DATE'] >= '2015-09-01') & (df_merged['DATE'] <= '2015-12-31')
df_merged.loc[mask, 'CNY'] = df_merged.loc[mask, 'CNY'].fillna(df_merged.loc[mask, 'USD'] / df_merged.loc[mask, 'CNY_USD'])
df_merged.head()
if 'CNY_USD' in df_merged.columns:
            df_merged.drop(columns=['CNY_USD'], inplace=True)
df_merged.to_csv('exchange.csv', index=False, encoding='utf-8-sig')

