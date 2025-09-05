import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_crude_oil_close_data():
    try:
        symbol = "CL=F"  # 원유 선물
        
        # 10년 전 날짜 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)

        print(f"원유 종가 데이터 다운로드 중...")
        print(f"심볼: {symbol}")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 원유 시세 다운로드
        crude_oil_data = yf.download(
            symbol, 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if crude_oil_data.empty:
            print("원유 데이터를 가져올 수 없습니다.")
            return None
        
        # 종가만 추출
        crude_oil_close = crude_oil_data['Close']
        crude_oil_close.reset_index(inplace=True)
        crude_oil_close.columns = ['날짜', '원유_종가']

        # CSV 파일로 저장
        filename = f"crude_oil_close_10years.csv"
        crude_oil_close.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"원유 종가 데이터 저장 완료!")
        print(f"파일명: {filename}")
        print(f"데이터 개수: {len(crude_oil_close)}개")

        return crude_oil_close

    except Exception as e:
        print(f"원유 데이터 다운로드 중 오류: {e}")
        return None

def download_brent_oil_data():
    """
    브렌트 오일 데이터만 다운로드하고 CSV로 저장
    """
    try:
        symbol = "BZ=F"  
        
        # 10년 전 날짜 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)

        print(f"\n브렌트 오일 데이터 다운로드 중...")
        print(f"심볼: {symbol}")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 브렌트 오일 데이터 다운로드
        brent_oil_data = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if brent_oil_data.empty:
            print("브렌트 오일 데이터를 가져올 수 없습니다.")
            return None

        # 종가만 추출 (브렌트 오일은 종가가 지수값)
        brent_oil_close = brent_oil_data['Close']
        brent_oil_close.reset_index(inplace=True)
        brent_oil_close.columns = ['날짜', '브렌트_오일_종가']

        # CSV 파일로 저장
        filename = f"brent_oil_close_10years.csv"
        brent_oil_close.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"✅ 브렌트 오일 데이터 저장 완료!")
        print(f"파일명: {filename}")
        print(f"데이터 개수: {len(brent_oil_close)}개")
        print(f"미리보기:")
        print(brent_oil_close.head())

        return brent_oil_close

    except Exception as e:
        print(f"브렌트 오일 데이터 다운로드 중 오류: {e}")
        return None

if __name__ == "__main__":
    # 원유 종가 데이터 다운로드
    crude_oil_data = download_crude_oil_close_data()

    # 브렌트 오일 데이터 다운로드
    brent_oil_data = download_brent_oil_data()

    print(f"\n🎉 모든 다운로드 완료!")
    print(f"생성된 파일:")
    print(f"- crude_oil_close_10years.csv (원유 종가)")
    print(f"- brent_oil_close_10years.csv (브렌트 오일 종가)")