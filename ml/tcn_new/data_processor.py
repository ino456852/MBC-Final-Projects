import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 상위 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent
ml_dir = current_dir.parent
if str(ml_dir) not in sys.path:
    sys.path.append(str(ml_dir))

from ml.data_merge import create_merged_dataset
from .constant import (
    PRED_TRUE_DIR,
    LOOK_BACK,
    PRED_TRUE_CSV,
    SCALER_DIR,
)
import pandas as pd


class DataProcessor:
    def __init__(self):
        self.origin_data = create_merged_dataset()
        self.origin_data["kr_us_diff"] = (
            self.origin_data["kr_rate"] - self.origin_data["us_rate"]
        )
        self.origin_data["dgs_jpy_diff"] = (
            self.origin_data["dgs10"] - self.origin_data["jpy10"]
        )
        self.origin_data["dgs_eur_diff"] = (
            self.origin_data["dgs10"] - self.origin_data["eur10"]
        )
        self.targets = ["usd", "cny", "jpy", "eur"]

    def add_indicators(self, data: pd.DataFrame, target: str):
        periods = [5, 10, 20]  # 기간 줄임
        for p in periods:
            data[f"MA_{p}"] = data[target].rolling(p).mean()
            data[f"EMA_{p}"] = data[target].ewm(span=p, adjust=False).mean()
            
        # 볼린저 밴드 추가
        data['BB_upper'] = data[f'MA_20'] + (data[target].rolling(20).std() * 2)
        data['BB_lower'] = data[f'MA_20'] - (data[target].rolling(20).std() * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        
        # RSI 추가
        delta = data[target].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data.dropna(inplace=True)

    def get_target_scaler(self, target: str):
        scaler_path = SCALER_DIR / f"{target}_target_scaler.pkl"
        return joblib.load(scaler_path)

    def get_feature_scaler(self, target: str):
        scaler_path = SCALER_DIR / f"{target}_feature_scaler.pkl"
        return joblib.load(scaler_path)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_len: int, y_index=None
    ):
        Xs, ys, idxs = [], [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
            if y_index is not None:
                idxs.append(y_index[i + seq_len])
        if y_index is not None:
            return np.array(Xs), np.array(ys), np.array(idxs)
        return np.array(Xs), np.array(ys)

    def save_predictions_csv(self, y_true, y_pred, target: str, index):
        """
        실제값과 예측값을 CSV로 저장합니다.
        """
        # 역변환
        scaler = self.get_target_scaler(target=target)
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        df = pd.DataFrame({"true": y_true_inv, "pred": y_pred_inv})
        df.index = index

        csv_file = f"{target}_{PRED_TRUE_CSV}"
        os.makedirs(PRED_TRUE_DIR, exist_ok=True)
        df.to_csv(PRED_TRUE_DIR / csv_file)
        print(f"Saved: {csv_file}")

    def get_proceed_data(self, target) -> pd.DataFrame:
        data = self.origin_data.copy()

        if target == "usd":
            keep_cols = ["dgs10", "vix", "dxy", "kr_us_diff", "kr_rate", "us_rate", "usd"]
        elif target == "cny":
            keep_cols = ["cny_fx_reserves", "cny_trade_bal", "wti", "vix", "cny"]
        elif target == "jpy":
            keep_cols = ["jpy10", "dgs10", "dgs_jpy_diff", "vix", "jpy"]
        elif target == "eur":
            keep_cols = ["eur10", "dxy", "dgs_eur_diff", "vix", "eur"]

        data = data[keep_cols]
        self.add_indicators(data=data, target=target)

        return data

    def get_sequence_data(self, target: str):
        data = self.get_proceed_data(target=target)

        X = data.drop(columns=[target])
        y = data[target]

        # 각 특성별로 별도의 스케일러 적용
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler = MinMaxScaler(feature_range=(0, 1))

        # 타겟과 피처를 분리해서 스케일링
        target_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        features_scaled = feature_scaler.fit_transform(X.values)

        # scaler를 파일로 저장
        os.makedirs(SCALER_DIR, exist_ok=True)
        joblib.dump(
            target_scaler, os.path.join(SCALER_DIR, f"{target}_target_scaler.pkl")
        )
        joblib.dump(
            feature_scaler, os.path.join(SCALER_DIR, f"{target}_feature_scaler.pkl")
        )

        # 인덱스도 같이 넘김
        X_seq, y_seq, y_idxs = self.create_sequences(
            X=features_scaled, y=target_scaled, seq_len=LOOK_BACK, y_index=y.index
        )

        return X_seq, y_seq, y_idxs
