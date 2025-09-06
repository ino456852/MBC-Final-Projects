import pandas as pd
import numpy as np

# --- 설정 ---
# 읽어올 원본 데이터 파일 이름
INPUT_FILENAME = 'currency.csv'
# 계산 결과를 저장할 새 파일 이름
OUTPUT_FILENAME = 'final_calculated_data_cleaned.csv' # 결과 파일 이름 변경

def calculate_rate_from_local_file():
    """
    로컬에 저장된 currency.csv 파일을 읽어 KRW/CNY 환율을 계산하고
    보조 컬럼을 삭제한 뒤 최종 저장합니다.
    """
    try:
        # 1. 로컬 CSV 파일 불러오기
        print(f"로컬 파일 '{INPUT_FILENAME}'을 불러옵니다...")
        df = pd.read_csv(INPUT_FILENAME)
        
        # 2. DATE 컬럼을 날짜 형식으로 변환하고 인덱스로 설정
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        print("파일 로드 완료. 환율 계산을 시작합니다...")

        # 3. 교차환율 계산
        if 'USD' not in df.columns or 'CNY_USD' not in df.columns:
            print("❌ 오류: 파일에 'USD' 또는 'CNY_USD' 컬럼이 없습니다. 원본 파일을 확인해주세요.")
            return

        calculated_values = np.where(
            df['CNY_USD'].notna(),
            (df['USD'] / df['CNY_USD']).round(2),
            np.nan
        )
        
        df['CNY'].fillna(pd.Series(calculated_values, index=df.index), inplace=True)
        
        # 4. 결과 확인
        print("\n✅ 계산이 완료되었습니다.")
        print("'CNY' 컬럼이 계산된 원/위안 환율로 업데이트되었습니다.")
        
        print("\n--- 계산 결과 미리보기 (CNY_USD가 있는 부분) ---")
        print(df[df['CNY_USD'].notna()][['USD', 'CNY_USD', 'CNY']].head())
        
        # =============================================================
        # 5. 저장 전, 보조적으로 사용된 'CNY_USD' 컬럼 삭제 (추가된 부분)
        # =============================================================
        if 'CNY_USD' in df.columns:
            df.drop(columns=['CNY_USD'], inplace=True)
            print("\n✅ 계산에 사용된 'CNY_USD' 보조 컬럼을 삭제했습니다.")
        # =============================================================

        # 6. 변경된 전체 데이터를 새 파일로 저장
        df.to_csv(OUTPUT_FILENAME, encoding='utf-8-sig')
        print(f"\n✅ 최종 결과가 '{OUTPUT_FILENAME}' 파일로 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_FILENAME}' 파일을 찾을 수 없습니다. 스크립트와 같은 폴더에 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 알 수 없는 오류가 발생했습니다: {e}")

# 스크립트 실행
if __name__ == "__main__":
    calculate_rate_from_local_file()