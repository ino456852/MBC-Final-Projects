import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_krw_currencies():
    """
    KRW 기준 환율 다운로드 (위안만 계산, 나머지는 직접)
    """
    # 직접 다운로드 가능한 KRW 페어들
    direct_pairs = {
        "USDKRW=X": "USD_KRW",  # 달러/원
        "EURKRW=X": "EUR_KRW",  # 유로/원  
        "JPYKRW=X": "JPY_KRW",  # 엔/원
        "GBPKRW=X": "GBP_KRW",  # 파운드/원
        "CNYKRW=X": "CNY_KRW"   # 위안/원
    }

    
    # 10년 전 날짜 계산
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    print(f"KRW 기준 환율 데이터 다운로드")
    print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print("="*50)
    
    successful_downloads = []
    
    # 직접 다운로드
    for symbol, currency_name in direct_pairs.items():
        try:
            currency_data = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )
            
            if currency_data.empty:
                print(f"{currency_name} 데이터를 가져올 수 없습니다.")
                continue
            
            df = pd.DataFrame()
            df['날짜'] = currency_data.index.strftime('%Y-%m-%d')
            df[f'{currency_name}_환율'] = currency_data['Close'].values
            
            # CSV 파일로 저장
            filename = f"{currency_name}_10years.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"{currency_name} 저장 완료")
            print(f"파일명: {filename}")
            print(f"데이터 개수: {len(df)}개")
            
            successful_downloads.append(currency_name)
            
        except Exception as e:
            print(f"{currency_name} 다운로드 중 오류: {e}")
            continue
    

if __name__ == "__main__":
    download_krw_currencies()