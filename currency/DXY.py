from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta

# FRED API 초기화
fred = Fred(api_key='3bea88d926222f495682fc69abc96b80')

# 달러 인덱스 데이터 가져오기
dollar_index = fred.get_series('DTWEXBGS', start='2015-08-31', end='2025-08-22')

# CSV 파일로 저장 (기본)
dollar_index.to_csv('dollar_index.csv')

# 헤더와 함께 저장
dollar_index.to_csv('dollar_index.csv', header=['Dollar_Index']) 