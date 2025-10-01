import json
import os
import sys
from pathlib import Path
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import tensorflow as tf

# 현재 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from model import build_model
from constant import (
    BASE_DIR,
    KERAS_FILE_TEMPLATE,
    LOOK_BACK,
    MODEL_DIR,
)
from .data_processor import DataProcessor


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def rolling_split_index(total_len, train_size=800, test_size=200):
    """
    전체 데이터 길이에서 rolling window 방식으로
    train/test 인덱스 배열을 생성합니다.
    """
    for start in range(0, total_len - train_size - test_size + 1, test_size):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )


def get_best_fold(X_train, y_train, X_test, y_test):
    hyperparams_candidates = {
        "tcn_filters": [32, 64],
        "dropout_rate": [0.1, 0.2],
        "learning_rate": [0.001, 0.0005],
    }

    best_val_loss = float('inf')
    best_params = None

    for tcn_filters, dropout_rate, learning_rate in product(
        hyperparams_candidates["tcn_filters"],
        hyperparams_candidates["dropout_rate"],
        hyperparams_candidates["learning_rate"],
    ):
        model = build_model(
            LOOK_BACK, X_train.shape[2], tcn_filters, dropout_rate, learning_rate
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
            )
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )
        val_loss = min(history.history["val_loss"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (tcn_filters, dropout_rate, learning_rate)

    return best_params, best_val_loss


def train():
    data_processor = DataProcessor()
    targets = data_processor.targets

    results = {}

    for target in targets:
        X_seq, y_seq, y_idxs = data_processor.get_sequence_data(target=target)

        best_params = tuple()
        best_val_loss = float('inf')
        last_test_idx = None

        total_len = len(X_seq) + LOOK_BACK
        print(f"{target.upper()} 모델 학습 시작: 데이터갯수: {total_len}")
        for train_idx, test_idx in rolling_split_index(total_len):
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]
            y_test_idx = y_idxs[test_idx]
            last_test_idx = (X_test, y_test, y_test_idx)

            fold_best_params, fold_best_val_loss = get_best_fold(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )

            if fold_best_val_loss < best_val_loss:
                best_params = fold_best_params

        print(f"{target.upper()} 최적 하이퍼파라미터: {best_params}")

        model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        model.fit(
            X_seq,
            y_seq,
            shuffle=False,
            epochs=100,
            batch_size=16,
            callbacks=callbacks,
            verbose=1,
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_name = f"{target}{KERAS_FILE_TEMPLATE}"
        model.save(os.path.join(MODEL_DIR, f"{model_name}"))

        X_test, y_test, y_test_idx = last_test_idx
        y_pred = model.predict(X_test)
        data_processor.save_predictions_csv(
            y_test, y_pred, target=target, index=y_test_idx
        )

        target_scaler = data_processor.get_target_scaler(target)
        y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        metrics = evaluate_predictions(y_test_inv, y_pred_inv)

        best_params_dict = {
            "tcn_filters": best_params[0],
            "dropout_rate": best_params[1],
            "learning_rate": best_params[2],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

    with open(os.path.join(BASE_DIR, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    train()
