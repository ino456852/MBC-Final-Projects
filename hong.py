import yfinance as yf
import pandas_datareader.data as web
import datetime
import pandas as pd
import requests

vix = yf.download('^VIX', start='2015-09-01', interval='1d')['Close']
dxy = yf.download('DX-Y.NYB', start='2015-09-01', interval='1d')['Close']
wti = yf.download('CL=F', start='2015-09-01', interval='1d')['Close']

# 미국 기준금리 (월별)
us_rate = web.DataReader('FEDFUNDS', 'fred', datetime.datetime(2015,9,1))

dgs10 = web.DataReader('DGS10', 'fred', datetime.datetime(2015,9,1))

vix.head() # 단위: 퍼센트 (S&P 500 옵션 가격으로 향후 30일간 예상 변동성 추정한 값)
dxy.head() # 단위: 지수 (1973.03 [고정환율 -> 변동환율 변환 시점] = 100 기준)
wti.head() # 단위: USD/배럴
us_rate.head() # 단위: 퍼센트
dgs10.head() # 단위: 퍼센트 (공휴일, 휴장일은 결측치)

# 한국 기준금리 (월별)
url = f"https://ecos.bok.or.kr/api/StatisticSearch/JLIMTOGUHFYU4IMR7A3T/json/kr/1/200/722Y001/M/201509/202509/0101000/"

response = requests.get(url)
data_json = response.json()
rows = data_json['StatisticSearch']['row']

kr_rate = pd.DataFrame(rows)
kr_rate = kr_rate[['TIME', 'DATA_VALUE']].rename(columns={'TIME':'DATE', 'DATA_VALUE':'kr_rate'})
kr_rate['DATE'] = pd.to_datetime(kr_rate['DATE'], format='%Y%m')
kr_rate['kr_rate'] = kr_rate['kr_rate'].astype(float)
kr_rate.head()

vix.to_csv('vix.csv', encoding='utf-8-sig')
dxy.to_csv('dxy.csv', encoding='utf-8-sig')
wti.to_csv('wti.csv', encoding='utf-8-sig')
us_rate.to_csv('us_rate.csv', encoding='utf-8-sig')
dgs10.to_csv('dgs10.csv', encoding='utf-8-sig')
kr_rate.to_csv('kr_rate.csv', index=False, encoding='utf-8-sig')

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

# CNY/USD (15.09.01 ~ 15.12.31)
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

df_merged.to_csv('currency.csv', index=False, encoding='utf-8-sig')