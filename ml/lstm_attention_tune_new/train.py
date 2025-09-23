import copy
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from .models import MODELS_TO_TRAIN
from .constant import (
    BASE_DIR,
    BEST_PARAMS_PTH,
    MODELS_DIR,
    LOOK_BACK,
    PRED_TRUE_CSV,
    PRED_TRUE_DIR,
)
from .data_processor import DataProcessor


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def train():
    if not os.path.exists(BEST_PARAMS_PTH):
        print(f"[{BEST_PARAMS_PTH}] 파일을 찾을 수 없습니다.")

    data_processor = DataProcessor()
    targets = data_processor.targets

    try:
        with open(os.path.join(BASE_DIR, "train_results.json"), "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    default_params = {
        "lstm_units": 150,
        "lstm_units_half": 75,
        "dropout_rate": 0.2,
        "num_heads": 8,
        "key_dim": 64,
        "learning_rate": 0.001,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PRED_TRUE_DIR, exist_ok=True)

    with open(BEST_PARAMS_PTH, "r") as f:
        best_params = json.load(f)

    for target in targets:
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, y_test_idxs = (
            data_processor.get_sequence_data(target=target)
        )

        params = copy.deepcopy(default_params)
        params.update(best_params[target])

        num_features = X_train_seq.shape[2]

        for model_name, build_fnc in MODELS_TO_TRAIN.items():
            save_path = MODELS_DIR / f"{target}_{model_name}_lstm.keras"

            if os.path.exists(save_path):
                print(f"{model_name} -> 이미 훈련된 모델 발견. 건너뜁니다.")

                continue

            print(f"{target}({model_name}) -> 모델 훈련 시작...")

            if model_name == "cross":
                num_base_features = data_processor.num_base_features
                model = build_fnc(
                    look_back=LOOK_BACK,
                    num_base_features=num_base_features,
                    num_features=num_features,
                    params=params,
                )
            else:
                model = build_fnc(
                    look_back=LOOK_BACK, num_features=num_features, params=params
                )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=save_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,  # 모델 전체 저장
                    verbose=0,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
            ]

            model.fit(
                X_train_seq,
                y_train_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1,  # 로그 출력 여부 0 or 1
            )

            y_pred = model.predict(X_test_seq)

            csv_path = PRED_TRUE_DIR / f"{target}_{model_name}_{PRED_TRUE_CSV}"
            data_processor.save_predictions_csv(
                y_test_seq, y_pred, target=target, filepath=csv_path, index=y_test_idxs
            )

            # === 역변환 후 성능지표 계산 ===
            target_scaler = data_processor.get_target_scaler(target)
            y_test_inv = target_scaler.inverse_transform(
                y_test_seq.reshape(-1, 1)
            ).flatten()
            y_pred_inv = target_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()

            metrics = evaluate_predictions(y_test_inv, y_pred_inv)
            results[target] = metrics

    # 결과를 JSON 파일로 저장
    with open(os.path.join(BASE_DIR, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    train()
