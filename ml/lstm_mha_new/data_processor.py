import os
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from ..data_merge import create_merged_dataset
from .constant import (
    LOOK_BACK,
    SCALER_DIR,
)
import pandas as pd


class DataProcessor:
    def __init__(self):
        self.origin_data = create_merged_dataset()
        self.origin_data["kr_us_diff"] = (
            self.origin_data["kr_rate"] - self.origin_data["us_rate"]
        )
        self.targets = ["usd", "cny", "jpy", "eur", "gbp"]

        # 현재 컬럼수 - 종속변수 갯수 - 1(kr_us_diff)
        self.num_base_features = self.origin_data.shape[1] - len(self.targets) - 1

    def add_indicators(self, data: pd.DataFrame, target: str):
        periods = [5, 20, 60, 120]
        for p in periods:
            data[f"MA_{p}"] = data[target].rolling(p).mean()
            data[f"EMA_{p}"] = data[target].ewm(span=p, adjust=False).mean()
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

    def save_predictions_csv(self, y_true, y_pred, target: str, filepath, index):
        """
        실제값과 예측값을 CSV로 저장합니다.
        """
        # 역변환
        scaler = self.get_target_scaler(target=target)
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        df = pd.DataFrame({"true": y_true_inv, "pred": y_pred_inv})
        df.index = index

        df.to_csv(filepath)
        print(f"Saved: {filepath}")

    def get_proceed_data(self, target) -> pd.DataFrame:
        data = self.origin_data.copy()

        self.add_indicators(data=data, target=target)

        return data

    def get_sequence_data(self, target: str):
        data = self.get_proceed_data(target=target)

        # TimeSeriesSplit을 이용해 마지막 split의 test set 추출
        tss = TimeSeriesSplit(n_splits=5)
        all_splits = list(tss.split(data))
        train_indices, test_indices = all_splits[-1]

        train_df = data.iloc[train_indices]
        test_df = data.iloc[test_indices]

        target_scaler = MinMaxScaler()
        feature_scaler = MinMaxScaler()

        features = list(data.columns)
        features.remove(target)
        train_features_scaled = feature_scaler.fit_transform(train_df[features])
        train_target_scaled = target_scaler.fit_transform(train_df[[target]])
        test_features_scaled = feature_scaler.transform(test_df[features])
        test_target_scaled = target_scaler.transform(test_df[[target]])

        # scaler를 파일로 저장
        os.makedirs(SCALER_DIR, exist_ok=True)
        joblib.dump(
            target_scaler, os.path.join(SCALER_DIR, f"{target}_target_scaler.pkl")
        )
        joblib.dump(
            feature_scaler, os.path.join(SCALER_DIR, f"{target}_feature_scaler.pkl")
        )

        X_train_seq, y_train_seq = self.create_sequences(
            X=train_features_scaled, y=train_target_scaled, seq_len=LOOK_BACK
        )
        X_test_seq, y_test_seq, y_test_idxs = self.create_sequences(
            X=test_features_scaled,
            y=test_target_scaled,
            seq_len=LOOK_BACK,
            y_index=test_df.index,
        )

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, y_test_idxs
