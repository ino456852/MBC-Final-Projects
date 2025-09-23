import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .constant import FEATURES, LOOK_BACK


def add_moving_average_features(
    df: pd.DataFrame, target: str, alphas=None
) -> pd.DataFrame:
    """
    지정한 타겟 컬럼에 대해 여러 기간의 이동평균(MA)과 지수이동평균(EMA) 피처를 추가합니다.
    alphas: EMA 계산에 사용할 alpha 값 딕셔너리 (key: 기간, value: alpha)
    """
    if alphas is None:
        alphas = {5: 0.1, 20: 0.1, 60: 0.1, 120: 0.1}

    for period in alphas.keys():
        # 단순 이동평균(MA) 피처 추가
        df[f"{target}_MA{period}"] = df[target].rolling(period).mean()
        # 지수 이동평균(EMA) 피처 추가
        df[f"{target}_EMA{period}"] = (
            df[target].ewm(alpha=alphas[period], adjust=False).mean()
        )
    return df


def create_sequences(features, targets, look_back):
    """
    시계열 데이터를 LSTM 입력 형태로 변환합니다.
    features: 입력 특성 배열 (2D)
    targets: 타겟 배열 (1D 또는 2D)
    look_back: 시퀀스 길이
    반환값: (X, y) - X는 (샘플수, look_back, 특성수), y는 (샘플수, 타겟수)
    """
    X = [features[i : i + look_back] for i in range(len(features) - look_back)]
    y = [targets[i + look_back] for i in range(len(features) - look_back)]
    return np.array(X), np.array(y)


def filter_dataframe(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """
    데이터프레임을 시작일자와 종료일자로 필터링합니다.
    start_date, end_date: 'YYYY-MM-DD' 형식 문자열 또는 None
    """
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def load_preprocess_data(target: str, alphas=None):
    """
    전체 데이터셋을 불러와서, 피처 생성/스케일링/시퀀스 변환까지 전처리 후 반환합니다.
    target: 예측할 타겟 컬럼명
    alphas: EMA 계산용 alpha 값 딕셔너리
    반환값: dict (df, features, feature_scaler, target_scaler, X_seq, y_seq)
    """
    from ml.data_merge import create_merged_dataset

    if alphas is None:
        alphas = {5: 0.1, 20: 0.1, 60: 0.1, 120: 0.1}
    df = create_merged_dataset()  # 데이터셋 생성 및 불러오기
    if df is None or df.empty:
        raise ValueError("데이터 로딩 실패")
    # 금리 차이 피처 생성
    df["kr_us_diff"] = df["kr_rate"] - df["us_rate"]
    # 타겟에 대한 MA/EMA 피처 생성
    df = add_moving_average_features(df, target, alphas)
    ma_ema_periods = [5, 20, 60, 120]
    # MA/EMA 피처명 리스트 생성
    ma_ema_features = [
        f"{target}_{kind}{p}" for kind in ["MA", "EMA"] for p in ma_ema_periods
    ]
    # 전체 사용 피처 리스트 (중복 제거)
    features = list(dict.fromkeys(FEATURES + ma_ema_features))
    # 결측치 제거 (모든 피처와 타겟에 대해)
    df = df.dropna(subset=features + [target])
    # 피처/타겟 스케일러 생성 및 스케일링
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(df[features])
    target_scaled = target_scaler.fit_transform(df[[target]])
    # LSTM 입력용 시퀀스 생성
    X_seq, y_seq = create_sequences(features_scaled, target_scaled, LOOK_BACK)
    return {
        "df": df,
        "features": features,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_seq": X_seq,
        "y_seq": y_seq,
    }
