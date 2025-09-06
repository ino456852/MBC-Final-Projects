import pandas as pd
import numpy as np

# --- 설정 ---
# 스크립트와 같은 폴더에 있는 입력 파일 이름
INPUT_FILENAME = 'currency.csv'
# 새로 저장될 결과 파일 이름
OUTPUT_FILENAME = 'currency_with_final_rate.csv'

def calculate_krw_cny_rate():
    """
    currency.csv 파일을 읽어 KRW/CNY 환율을 계산하고 결과를 저장합니다.
    """
    try:
        # 1. CSV 파일 불러오기
        df = pd.read_csv(INPUT_FILENAME)
        
        # 2. DATE 컬럼을 날짜 형식으로 변환하고 인덱스로 설정
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        print(f"✅ '{INPUT_FILENAME}' 파일을 성공적으로 로드했습니다.")

        # 3. 교차환율 계산
        # KRW/CNY = KRW/USD / CNY/USD
        # 'CNY' 컬럼에 계산 결과를 덮어씁니다.
        # np.where를 사용하여 CNY_USD가 비어있지 않은 경우에만 계산을 수행합니다.
        # 기존 CNY 컬럼의 값을 보존하면서, 계산 가능한 경우에만 업데이트합니다.
        
        # 계산된 값을 담을 새로운 시리즈 생성
        calculated_values = np.where(
            df['CNY_USD'].notna(),                          # 조건: CNY_USD 값이 비어있지 않으면
            (df['USD'] / df['CNY_USD']).round(2),            # 계산 수행: (USD 값 / CNY_USD 값)
            np.nan                                          # 조건이 거짓이면: 계산하지 않음 (NaN)
        )
        
        # 원래 CNY 컬럼의 값에 계산된 값을 덮어쓰기 (계산된 값이 있는 경우에만)
        df['CNY'].fillna(pd.Series(calculated_values, index=df.index), inplace=True)
        
        # 4. 결과 확인
        print("\n✅ 계산이 완료되었습니다.")
        print("'USD' 컬럼과 'CNY_USD' 컬럼을 사용하여 'CNY' 컬럼을 원/위안 환율로 업데이트했습니다.")
        
        # 계산에 사용된 주요 컬럼과 결과 확인
        print("\n--- 계산 결과 미리보기 (상위 5개) ---")
        print(df[['USD', 'CNY_USD', 'CNY']].head())
        
        print("\n--- 계산 결과 미리보기 (하위 5개) ---")
        print(df[['USD', 'CNY_USD', 'CNY']].tail())

        # 5. 변경된 전체 데이터를 새 파일로 저장
        df.to_csv(OUTPUT_FILENAME, encoding='utf-8-sig')
        print(f"\n✅ 모든 계산 결과가 '{OUTPUT_FILENAME}' 파일로 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_FILENAME}' 파일을 찾을 수 없습니다. 스크립트와 같은 폴더에 있는지 확인해주세요.")
    except KeyError as e:
        print(f"❌ 오류: 파일에서 필요한 컬럼({e})을 찾을 수 없습니다. 'USD', 'CNY_USD' 컬럼이 파일에 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 알 수 없는 오류가 발생했습니다: {e}")

# 스크립트 실행
if __name__ == "__main__":
    calculate_krw_cny_rate()