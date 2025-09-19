import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import (
    get_data, add_moving_averages, create_sequences, scale_data,
    TARGETS, BASE_FEATURES, LOOK_BACK
)
from .model import build_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_predict_future(best_params=None):
    if best_params is None:
        best_params = {
            'units': 128, 'lr': 0.001, 'dropout': 0.3, 'bidirectional': True
        }

    data = get_data()
    all_predictions = {}
    seq_len = LOOK_BACK
    split_date = '2024-01-01'

    for target in TARGETS:
        print(f"\n[{target.upper()}] 모델 학습 시작...")

        target_data = data.copy()
        target_data = add_moving_averages(target_data, target)
        features = BASE_FEATURES + [f'{target}_MA{p}' for p in [5, 20, 60, 120]]
        subset = target_data[features + [target]]

        train_data = subset[subset.index < split_date]
        test_data = subset[subset.index >= split_date]

        X_train, y_train, X_test, y_test, scaler_X, scaler_y = scale_data(train_data, test_data, features, target)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)

        model = build_model(seq_len, len(features), best_params)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True, verbose=0
        )

        model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

        # 저장
        model.save(os.path.join(MODEL_DIR, f'model_{target}.keras'))
        with open(os.path.join(MODEL_DIR, f'scaler_{target}.pkl'), 'wb') as f:
            pickle.dump({'feature_scaler': scaler_X, 'target_scaler': scaler_y, 'features': features}, f)

        # 예측
        predictions = []
        current_sequence = X_train[-seq_len:]

        for i in range(len(test_data)):
            pred_scaled = model.predict(current_sequence.reshape(1, seq_len, len(features)), verbose=0)
            pred_value = scaler_y.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred_value)

            if i < len(X_test) - 1:
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = X_test[i]

        all_predictions[target] = {
            'dates': test_data.index,
            'actual': y_test,
            'predicted': predictions
        }

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    result_df = pd.DataFrame(index=test_data.index)
    for target in TARGETS:
        result_df[f'Actual_{target.upper()}'] = all_predictions[target]['actual']
        result_df[f'Predicted_{target.upper()}'] = all_predictions[target]['predicted']

    csv_path = os.path.join(MODEL_DIR, '2024_predictions.csv')
    result_df.to_csv(csv_path)
    print(f"\n✅ 예측 결과 저장 완료: {csv_path}")

    return result_df


if __name__ == "__main__":
    print("2024년 환율 예측 모델 학습 시작")
    train_and_predict_future()
