import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_gold_close_data():
    try:
        symbol = "GC=F"  # 금 선물
        
        # 10년 전 날짜 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        
        print(f"금 종가 데이터 다운로드 중...")
        print(f"심볼: {symbol}")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 금 시세 다운로드
        gold_data = yf.download(
            symbol, 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if gold_data.empty:
            print("금 데이터를 가져올 수 없습니다.")
            return None
        
        # 종가만 추출
        gold_close = gold_data['Close']
        gold_close.reset_index(inplace=True)
        gold_close.columns = ['날짜', '금_종가']
        
        # CSV 파일로 저장
        filename = f"gold_close_10years.csv"
        gold_close.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"금 종가 데이터 저장 완료!")
        print(f"파일명: {filename}")
        print(f"데이터 개수: {len(gold_close)}개")
        
        return gold_close
        
    except Exception as e:
        print(f"금 데이터 다운로드 중 오류: {e}")
        return None

def download_vix_data():
    """
    VIX 지수 데이터만 다운로드하고 CSV로 저장
    """
    try:
        symbol = "^VIX"  # VIX 변동성 지수
        
        # 10년 전 날짜 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        
        print(f"\nVIX 지수 데이터 다운로드 중...")
        print(f"심볼: {symbol}")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # VIX 데이터 다운로드
        vix_data = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if vix_data.empty:
            print("VIX 데이터를 가져올 수 없습니다.")
            return None
        
        # 종가만 추출 (VIX는 종가가 지수값)
        vix_close = vix_data['Close']
        vix_close.reset_index(inplace=True)
        vix_close.columns = ['날짜', 'VIX_지수']
        
        # CSV 파일로 저장
        filename = f"vix_10years.csv"
        vix_close.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ VIX 지수 데이터 저장 완료!")
        print(f"파일명: {filename}")
        print(f"데이터 개수: {len(vix_close)}개")
        print(f"미리보기:")
        print(vix_close.head())
        
        return vix_close
        
    except Exception as e:
        print(f"VIX 데이터 다운로드 중 오류: {e}")
        return None

if __name__ == "__main__":
    # 금 종가 데이터 다운로드
    gold_data = download_gold_close_data()
    
    # VIX 지수 데이터 다운로드  
    vix_data = download_vix_data()
    
    print(f"\n🎉 모든 다운로드 완료!")
    print(f"생성된 파일:")
    print(f"- gold_close_10years.csv (금 종가)")
    print(f"- vix_10years.csv (VIX 지수)")