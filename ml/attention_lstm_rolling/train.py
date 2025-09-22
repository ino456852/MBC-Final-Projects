import os
import numpy as np
from itertools import product
from constant import TARGETS, LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE
from preprocess import load_preprocess_data
from model import build_model
import tensorflow as tf


def rolling_split_indices(total_len, train_size=1200, test_size=300):
    for start in range(0, total_len - train_size - test_size + 1, test_size):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )


def train_and_save_models(alphas=None):
    if alphas is None:
        alphas = {5: 0.1, 20: 0.1, 60: 0.1, 120: 0.1}
    HYPERPARAMS_CANDIDATES = {
        "lstm_units": [100, 150],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
    }
    all_models_info = {}
    for target in TARGETS:
        print(f"= {target.upper()} 모델 학습 시작 =")
        data = load_preprocess_data(target, alphas)
        df = data["df"]
        features = data["features"]
        X_seq = data["X_seq"]
        y_seq = data["y_seq"]
        val_losses_per_param = {}
        total_len = len(df)
        for train_idx, test_idx in rolling_split_indices(total_len):
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]
            best_fold_val_loss = float("inf")
            best_fold_params = None
            for lstm_units, dropout_rate, learning_rate in product(
                HYPERPARAMS_CANDIDATES["lstm_units"],
                HYPERPARAMS_CANDIDATES["dropout_rate"],
                HYPERPARAMS_CANDIDATES["learning_rate"],
            ):
                model = build_model(
                    LOOK_BACK, X_train.shape[2], lstm_units, dropout_rate, learning_rate
                )
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0,
                )
                val_loss = min(history.history["val_loss"])
                if val_loss < best_fold_val_loss:
                    best_fold_val_loss = val_loss
                    best_fold_params = (lstm_units, dropout_rate, learning_rate)
            val_losses_per_param.setdefault(best_fold_params, []).append(
                best_fold_val_loss
            )
        avg_val_losses = {
            params: np.mean(losses) for params, losses in val_losses_per_param.items()
        }
        best_params = min(avg_val_losses, key=avg_val_losses.get)
        print(
            f"{target.upper()} 최적 하이퍼파라미터: {best_params}, 평균 val_loss={avg_val_losses[best_params]:.4f}"
        )
        # 전체 데이터로 다시 fit
        final_model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        final_model.fit(
            X_seq, y_seq, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0
        )
        model_name = f"{target}{KERAS_FILE_TEMPLATE}"
        final_model.save(os.path.join(MODEL_DIR, f"{model_name}"))
        all_models_info[target] = {
            "model": final_model,
            "feature_scaler": data["feature_scaler"],
            "target_scaler": data["target_scaler"],
            "features": features,
        }
    return all_models_info


if __name__ == "__main__":
    print("학습 시작")
    train_and_save_models()
    print("학습 완료")
