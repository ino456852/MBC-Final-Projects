import tensorflow as tf
import numpy as np
import json, os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from .constant import (BASE_DIR, KERAS_FILE_TEMPLATE, LOOK_BACK, MODEL_DIR) 
from .data_processor import DataProcessor
from .model import build_model
from itertools import product

# 성능지표 계산
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

# rolling window 방식으로 학습/테스트 인덱스 생성 함수
def rolling_split_index(total_len, train_size=1200, test_size=300, step=300):
    for start in range(0, total_len - train_size - test_size + 1, step):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )

def train():
    data_processor = DataProcessor()
    targets = data_processor.targets
    all_results = {}

    # 하이퍼파라미터 후보
    HYPERPARAMS_CANDIDATES = {
        "lstm_units": [100, 150],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64]
    }
    
    all_param_combinations = list(product(
        HYPERPARAMS_CANDIDATES["lstm_units"],
        HYPERPARAMS_CANDIDATES["dropout_rate"],
        HYPERPARAMS_CANDIDATES["learning_rate"],
        HYPERPARAMS_CANDIDATES["batch_size"]
    ))
    
    for target in targets:
        print(f"\n 학습 시작: {target.upper()}")
        X_seq, y_seq, y_idxs = data_processor.get_sequence_data(target=target)
        total_len_seq = len(X_seq)
        train_size = 1200 
        test_size = 300 
        val_losses_per_param = {} 
        
        y_pred_all, y_true_all, y_idxs_all = [], [], []
        
        splits = list(rolling_split_index(total_len_seq, train_size, test_size, step=300))[:5]
        
        if not splits:
             print(f"데이터 부족으로 {target} 학습 건너뜀")
             all_results[target] = None 
             continue
        
        # 스케일러 미리 로드
        target_scaler = data_processor.get_target_scaler(target)
        best_params = None
        best_model_weights = None
        val_losses_per_param = {}

        # Rolling Window 교차 검증을 통한 최적 하이퍼파라미터 탐색
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"\n Fold {i+1}/{len(splits)} 시작")
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]
            y_test_idx = y_idxs[test_idx].flatten()
            
            if i == 0:
                best_val_loss = float("inf")
                best_fold_model = None
                for params in all_param_combinations:
                    lstm_units, dropout_rate, learning_rate, batch_size = params
                    model = build_model(LOOK_BACK, X_train.shape[2], lstm_units, dropout_rate, learning_rate)
                    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

                    print(f" - 하이퍼파라미터 테스트: {params}")
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=30, batch_size=batch_size,
                        callbacks=[early_stop],
                        shuffle=False, verbose=1
                    )

                    val_loss = min(history.history["val_loss"])
                    val_losses_per_param.setdefault(params, []).append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = params
                        best_fold_model = model
                
                print(f"최적 하이퍼파라미터: {best_params}, val_loss={best_val_loss:.4f}")
                model = best_fold_model
            else:
                print(f"최적 하이퍼파라미터로 모델 학습")
                lstm_units, dropout_rate, learning_rate, batch_size = best_params
                model = build_model(LOOK_BACK, X_train.shape[2], lstm_units, dropout_rate, learning_rate)
            if i > 0:
                model.fit(X_train, y_train, epochs=5, batch_size=batch_size, shuffle=False, verbose=1)
            y_pred = model.predict(X_test, verbose=1)
            y_pred_all.append(y_pred)
            y_true_all.append(y_test)
            y_idxs_all.append(y_test_idx)

        # 전체 예측 평가
        if y_pred_all:
            y_pred_concat = np.concatenate(y_pred_all)
            y_true_concat = np.concatenate(y_true_all)
            y_idxs_concat = np.concatenate(y_idxs_all)

            y_true_inv = target_scaler.inverse_transform(y_true_concat.reshape(-1, 1)).flatten()
            y_pred_inv = target_scaler.inverse_transform(y_pred_concat.reshape(-1, 1)).flatten()

            metrics = evaluate_predictions(y_true_inv, y_pred_inv)
            print(f"\n{target.upper()} 성능 평가 → RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")

            data_processor.save_predictions_csv(y_true_concat, y_pred_concat, target=target, index=y_idxs_concat)
            print(f"예측 결과 저장 완료: {target}_pred_true.csv")
        else:
            metrics = {"rmse": np.nan, "r2": np.nan, "mape": np.nan}
            print("예측 결과 없음. 평가 생략.")

        # 전체 데이터로 최종 모델 학습 및 저장
        print("최종 모델 전체 데이터로 학습")
        lstm_units, dropout_rate, learning_rate, batch_size = best_params
        final_model = build_model(LOOK_BACK, X_seq.shape[2], lstm_units, dropout_rate, learning_rate)
        final_model.fit(X_seq, y_seq, epochs=30, batch_size=batch_size, shuffle=False, verbose=1)

        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save(os.path.join(MODEL_DIR, f"{target}{KERAS_FILE_TEMPLATE}"))
        print(f"모델 저장 완료: {target}{KERAS_FILE_TEMPLATE}")

        scaler_info = {
            "min": target_scaler.min_.tolist(),
            "scale": target_scaler.scale_.tolist(),
            "data_min": target_scaler.data_min_.tolist(),
            "data_max": target_scaler.data_max_.tolist(),
            "feature_range": target_scaler.feature_range
        }

        all_results[target] = {
            "best_params": {
                "lstm_units": best_params[0],
                "dropout_rate": best_params[1],
                "learning_rate": best_params[2],
                "batch_size": best_params[3]
            },
            "metrics": metrics,
            "scaler": scaler_info
        }

    with open(os.path.join(BASE_DIR, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\n전체 학습 및 저장 완료")
    return all_results

if __name__ == "__main__":
    train()