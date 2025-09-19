import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
base_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(base_path)  # 프로젝트 루트 폴더
ml_path = os.path.join(base_path, 'ml')
for p in [base_path, ml_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.data_merge import create_merged_dataset

# 상수 정의
TARGETS = ['usd', 'cny', 'jpy', 'eur', 'gbp']
BASE_FEATURES = ['dxy', 'wti', 'vix', 'dgs10', 'kr_rate', 'us_rate', 'kr_us_diff']
LOOK_BACK = 60
MODEL_DIR = os.path.join(project_root, 'models')  # 프로젝트 루트의 models 폴더
os.makedirs(MODEL_DIR, exist_ok=True)

# 데이터 준비 함수
def get_data():
    df = create_merged_dataset()
    if df is None or df.empty:
        raise ValueError("데이터 로딩 실패")
    df['kr_us_diff'] = df['kr_rate'] - df['us_rate']
    missing = [c for c in TARGETS + BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"누락된 컬럼: {missing}")
    return df

# Attention 레이어 (원본과 동일하게 유지)
class CustomAttention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(input_shape[-1], activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1)
        super().build(input_shape)
    
    def call(self, x):
        a = tf.nn.softmax(self.dense2(self.dense1(x)), axis=1)
        return tf.reduce_sum(x * a, axis=1)

# 모델 생성 (원본 방식 유지)
def build_model(seq_len, n_features, params):
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    
    if params.get('bidirectional', True):
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(params['units'], return_sequences=True)
        )(inputs)
    else:
        lstm = tf.keras.layers.LSTM(params['units'], return_sequences=True)(inputs)
    
    attn = CustomAttention()(lstm)
    drop = tf.keras.layers.Dropout(params['dropout'])(attn)
    output = tf.keras.layers.Dense(1)(drop)
    
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(params['lr']), loss='mse')
    return model

# 시퀀스 생성 (원본 방식 유지)
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

# Optuna 최적화 함수 (원본 기능 유지)
def objective(trial):
    # 하이퍼파라미터 샘플링 (optimizer 제외)
    params = {
        'units': trial.suggest_int('units', 64, 256, step=32),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False])
    }
    
    seq_len = LOOK_BACK
    total_rmse = 0
    
    # 모든 통화에 대해 학습하고 평균 RMSE 계산 (검증용)
    for target in TARGETS:
        try:
            data = get_data()
            
            periods = [5, 20, 60, 120]  # 이동평균 기간
            for p in periods:
                data[f'{target}_MA{p}'] = data[target].rolling(p).mean()

            data.dropna(inplace=True)
            
            features = BASE_FEATURES + [f'{target}_MA{p}' for p in periods]
            subset = data[features + [target]]
            
            # 검증을 위해 최근 20% 데이터를 테스트용으로 분리
            split_idx = int(len(subset) * 0.8)
            train_data = subset.iloc[:split_idx]
            test_data = subset.iloc[split_idx:]
            
            # 스케일링
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_train = scaler_X.fit_transform(train_data[features])
            y_train = scaler_y.fit_transform(train_data[[target]])
            X_test = scaler_X.transform(test_data[features])
            y_test = test_data[target].values
            
            # 시퀀스 생성
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
            X_test_seq, _ = create_sequences(X_test, np.zeros(len(X_test)), seq_len)
            
            if len(X_test_seq) == 0:
                total_rmse += 1000.0
                continue
            
            # 모델 학습
            model = build_model(seq_len, len(features), params)
            
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
            )
            
            model.fit(
                X_train_seq, y_train_seq,
                epochs=30, batch_size=32, validation_split=0.1,
                callbacks=[early_stop], verbose=0
            )
            
            # 예측 및 평가
            pred_scaled = model.predict(X_test_seq, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled).flatten()
            actual = y_test[seq_len:]
            
            if len(actual) > 0:
                rmse = np.sqrt(mean_squared_error(actual, pred))
                total_rmse += rmse
            else:
                total_rmse += 1000.0
            
            # 메모리 정리
            del model
            tf.keras.backend.clear_session()
            
        except Exception:
            total_rmse += 1000.0
    
    return total_rmse / len(TARGETS)

def optimize_params(n_trials=50):
    """하이퍼파라미터 최적화"""
    print("하이퍼파라미터 최적화 시작...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"최적화 완료 최고 성능: {study.best_value:.4f}")
    print("최적 파라미터:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params

def predict_future_values(model, last_sequence, feature_scaler, target_scaler, days=1):
    """미래 값 예측 (1일 예측)"""
    # 1일만 예측
    pred_scaled = model.predict(last_sequence.reshape(1, *last_sequence.shape), verbose=0)
    pred_value = target_scaler.inverse_transform(pred_scaled)[0, 0]
    
    return [pred_value]

def train_and_predict_future(best_params=None, future_days=1):
    """2023년까지 학습하고 2024년 1월부터 현재까지 예측 (원본 기능 유지)"""
    if best_params is None:
        best_params = {
            'units': 128, 'lr': 0.001, 'dropout': 0.3, 'bidirectional': True
        }
    
    print("2023년까지 학습하고 2024년 1월부터 현재까지 예측...")
    
    data = get_data()
    all_predictions = {}
    seq_len = LOOK_BACK
    
    # 2024년 1월 1일 기준으로 데이터 분할
    split_date = '2024-01-01'
    
    for target in TARGETS:
        print(f"{target.upper()} 모델 학습 및 예측...")
        
        # 데이터 준비
        target_data = data.copy()
        ma5 = f'{target}_MA5'
        ma20 = f'{target}_MA20'
        ma60 = f'{target}_MA60'
        ma120 = f'{target}_MA120'
        
        target_data[ma5] = target_data[target].rolling(5).mean()
        target_data[ma20] = target_data[target].rolling(20).mean()
        target_data[ma60] = target_data[target].rolling(60).mean()
        target_data[ma120] = target_data[target].rolling(120).mean()
        target_data.dropna(inplace=True)
        
        features = BASE_FEATURES + [ma5, ma20, ma60, ma120]
        subset = target_data[features + [target]]
        
        # 2024년 1월 1일 기준으로 분할
        train_data = subset[subset.index < split_date]
        test_data = subset[subset.index >= split_date]
        
        print(f"   학습 데이터: {len(train_data)}개 (~2023년)")
        print(f"   예측 데이터: {len(test_data)}개 (2024년~)")
        
        # 스케일링 (학습 데이터 기준)
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train = scaler_X.fit_transform(train_data[features])
        y_train = scaler_y.fit_transform(train_data[[target]])
        X_test = scaler_X.transform(test_data[features])
        
        # 시퀀스 생성 (학습용)
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        
        # 모델 학습
        model = build_model(seq_len, len(features), best_params)
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True, verbose=0
        )
        
        model.fit(
            X_train_seq, y_train_seq,
            epochs=100, batch_size=32,
            callbacks=[early_stop], verbose=0
        )
        
        # 모델 저장
        model.save(os.path.join(MODEL_DIR, f'model_{target}.keras'))
        print(f"   모델 저장: model_{target}.keras")
        
        # 스케일러 저장 (중요!)
        import pickle
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{target}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': scaler_X,
                'target_scaler': scaler_y,
                'features': features
            }, f)
        print(f"   스케일러 저장: scaler_{target}.pkl")
        
        # 2024년부터 순차적 예측
        predictions = []
        current_sequence = X_train[-seq_len:]  # 마지막 60일 시퀀스
        
        for i in range(len(test_data)):
            # 예측
            pred_scaled = model.predict(current_sequence.reshape(1, seq_len, len(features)), verbose=0)
            pred_value = scaler_y.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred_value)
            
            # 다음 시퀀스를 위해 실제 데이터로 시퀀스 업데이트
            if i < len(X_test) - 1:
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = X_test[i]
        
        # 결과 저장
        all_predictions[target] = {
            'dates': test_data.index,
            'actual': test_data[target].values,
            'predicted': predictions
        }
        
        # 성능 평가
        mae = mean_absolute_error(test_data[target].values, predictions)
        rmse = np.sqrt(mean_squared_error(test_data[target].values, predictions))
        r2 = r2_score(test_data[target].values, predictions)
        
        print(f"   2024년~ 성능: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # 예측 결과를 DataFrame으로 정리
    result_df = pd.DataFrame(index=test_data.index)
    for target in TARGETS:
        result_df[f'Actual_{target.upper()}'] = all_predictions[target]['actual']
        result_df[f'Predicted_{target.upper()}'] = all_predictions[target]['predicted']
    
    # 결과 저장
    csv_path = os.path.join(base_path, '2024_predictions.csv')
    result_df.to_csv(csv_path)
    print(f"2024년 예측 결과 저장: {csv_path}")
    
    # 결과 출력
    print("\n" + "="*60)
    print("2024년 환율 예측 결과")
    print("="*60)
    
    print(f"\n 예측 기간: {test_data.index[0].strftime('%Y-%m-%d')} ~ {test_data.index[-1].strftime('%Y-%m-%d')}")
    
    for target in TARGETS:
        actual = all_predictions[target]['actual']
        predicted = all_predictions[target]['predicted']
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        print(f"\n {target.upper()}:")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   최근 실제값: {actual[-1]:.2f}")
        print(f"   최근 예측값: {predicted[-1]:.2f}")
    
    return result_df

if __name__ == "__main__":
    print("환율 2024년 예측 모델")
    print("="*40)
    
    choice = input("1: 기본 파라미터로 예측 | 2: 최적화 후 예측\n선택: ")
    
    if choice == "2":
        n_trials = int(input("최적화 시도 횟수 (기본 30): ") or "30")
        best_params = optimize_params(n_trials)
        result_df = train_and_predict_future(best_params)
    else:
        result_df = train_and_predict_future()
    
    print("\n 2024년 예측 완료!")
    print(" 결과는 '2024_predictions.csv'에 저장되었습니다.")