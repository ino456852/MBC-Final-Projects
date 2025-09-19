import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ml.data_merge import create_merged_dataset

TARGETS = ['usd', 'cny', 'jpy', 'eur', 'gbp']
BASE_FEATURES = ['dxy', 'wti', 'vix', 'dgs10', 'kr_rate', 'us_rate', 'kr_us_diff']
LOOK_BACK = 60


def get_data() -> pd.DataFrame:
    """데이터 로딩 및 기본 파생변수 생성"""
    df = create_merged_dataset()
    if df is None or df.empty:
        raise ValueError("데이터 로딩 실패")
    df['kr_us_diff'] = df['kr_rate'] - df['us_rate']
    return df


def add_moving_averages(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """이동평균 피처 생성"""
    for p in [5, 20, 60, 120]:
        df[f'{target}_MA{p}'] = df[target].rolling(p).mean()
    df.dropna(inplace=True)
    return df


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """시퀀스 데이터 생성"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def scale_data(train_data: pd.DataFrame, test_data: pd.DataFrame, features: list, target: str):
    """스케일링 처리 후 반환"""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train_data[features])
    y_train = scaler_y.fit_transform(train_data[[target]])
    X_test = scaler_X.transform(test_data[features])
    y_test = test_data[target].values

    return X_train, y_train, X_test, y_test, scaler_X, scaler_y
