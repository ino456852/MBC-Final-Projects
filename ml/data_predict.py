import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# data_merge.py에서 함수 import
from data_merge import create_merged_dataset

class AttentionLayer(tf.keras.layers.Layer):
    """Custom Attention Layer for LSTM"""
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        
    def call(self, query, values):
        # query: (batch_size, hidden_units)
        # values: (batch_size, time_steps, hidden_units)
        
        # query를 모든 time step에 대해 확장
        query_with_time_axis = tf.expand_dims(query, 1)  # (batch_size, 1, hidden_units)
        
        # Attention score 계산
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        
        # Attention weights 계산 (softmax)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector 계산
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class SelfAttentionLayer(tf.keras.layers.Layer):
    """Self-Attention Layer for sequence modeling"""
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights

class ExchangeRateAttentionLSTMPredictor:
    def __init__(self, target_currency='usd', sequence_length=60, test_size=0.2, attention_type='standard'):
        """
        Attention 기반 LSTM 환율 예측 모델 초기화
        
        Args:
            target_currency (str): 예측할 통화 ('usd', 'eur', 'gbp', 'jpy', 'cny')
            sequence_length (int): LSTM 입력 시퀀스 길이
            test_size (float): 테스트 데이터 비율
            attention_type (str): 어텐션 타입 ('standard', 'self', 'both')
        """
        self.target_currency = target_currency
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.attention_type = attention_type
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = None
        self.attention_weights = None
        
    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("데이터를 로드하고 있습니다...")
        
        # 데이터 로드
        self.df = create_merged_dataset()
        if self.df is None:
            raise Exception("데이터를 로드할 수 없습니다.")
        
        # 결측값 처리
        self.df = self.df.dropna()
        
        print(f"데이터 형태: {self.df.shape}")
        print(f"사용 가능한 통화: {self.df.columns.tolist()}")
        
        # 타겟 변수 확인
        if self.target_currency not in self.df.columns:
            raise Exception(f"타겟 통화 '{self.target_currency}'가 데이터에 존재하지 않습니다.")
        
        return self.df
    
    def create_features(self):
        """피처 엔지니어링"""
        print("피처를 생성하고 있습니다...")
        
        # 이동평균 추가
        for currency in ['usd', 'eur', 'gbp', 'jpy', 'cny']:
            if currency in self.df.columns:
                self.df[f'{currency}_ma7'] = self.df[currency].rolling(window=7).mean()
                self.df[f'{currency}_ma30'] = self.df[currency].rolling(window=30).mean()
                self.df[f'{currency}_volatility'] = self.df[currency].rolling(window=20).std()
        
        # 변화율 추가
        for col in self.df.columns:
            if col not in ['date']:
                self.df[f'{col}_pct_change'] = self.df[col].pct_change()
                # RSI 계산 (Relative Strength Index)
                delta = self.df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                self.df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드 (타겟 통화에 대해)
        rolling_mean = self.df[self.target_currency].rolling(window=20).mean()
        rolling_std = self.df[self.target_currency].rolling(window=20).std()
        self.df[f'{self.target_currency}_bb_upper'] = rolling_mean + (rolling_std * 2)
        self.df[f'{self.target_currency}_bb_lower'] = rolling_mean - (rolling_std * 2)
        self.df[f'{self.target_currency}_bb_ratio'] = (self.df[self.target_currency] - rolling_mean) / (rolling_std * 2)
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = self.df[self.target_currency].ewm(span=12).mean()
        ema26 = self.df[self.target_currency].ewm(span=26).mean()
        self.df[f'{self.target_currency}_macd'] = ema12 - ema26
        self.df[f'{self.target_currency}_macd_signal'] = self.df[f'{self.target_currency}_macd'].ewm(span=9).mean()
        
        # 결측값 제거
        self.df = self.df.dropna()
        
        # 피처 컬럼 정의 (타겟 변수 제외)
        self.feature_columns = [col for col in self.df.columns if col != self.target_currency]
        
        print(f"총 피처 수: {len(self.feature_columns)}")
        
    def create_sequences(self, X, y):
        """LSTM을 위한 시퀀스 데이터 생성"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def prepare_train_test_data(self):
        """학습/테스트 데이터 준비"""
        print("학습/테스트 데이터를 준비하고 있습니다...")
        
        # X, y 분리
        X = self.df[self.feature_columns].values
        y = self.df[self.target_currency].values.reshape(-1, 1)
        
        # 데이터 정규화
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 시퀀스 생성
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled.flatten())
        
        # 학습/테스트 분할
        split_idx = int(len(X_seq) * (1 - self.test_size))
        
        self.X_train = X_seq[:split_idx]
        self.X_test = X_seq[split_idx:]
        self.y_train = y_seq[:split_idx]
        self.y_test = y_seq[split_idx:]
        
        print(f"학습 데이터: {self.X_train.shape}")
        print(f"테스트 데이터: {self.X_test.shape}")
    
    def build_attention_model(self, lstm_units=64, attention_units=128, dropout_rate=0.2, learning_rate=0.001):
        """Attention 기반 LSTM 모델 구축"""
        print(f"{self.attention_type} Attention LSTM 모델을 구축하고 있습니다...")
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, len(self.feature_columns)))
        
        if self.attention_type == 'standard':
            # Standard LSTM with Attention
            lstm1 = LSTM(lstm_units, return_sequences=True, return_state=True)(inputs)
            lstm_output, state_h, state_c = lstm1
            
            lstm2 = LSTM(lstm_units, return_sequences=True, return_state=True)(lstm_output)
            lstm_output2, state_h2, state_c2 = lstm2
            
            # Attention layer
            attention_layer = AttentionLayer(attention_units)
            context_vector, attention_weights = attention_layer(state_h2, lstm_output2)
            
            # Dropout
            x = Dropout(dropout_rate)(context_vector)
            
        elif self.attention_type == 'self':
            # Self-Attention with LSTM
            # Multi-head self-attention
            self_attention = SelfAttentionLayer(d_model=lstm_units, num_heads=8)
            attn_output, attention_weights = self_attention(inputs, inputs, inputs)
            
            # inputs projection to match attn_output shape
            inputs_proj = Dense(lstm_units)(inputs)
            attn_output = LayerNormalization()(attn_output + inputs_proj)
            
            # LSTM layers
            lstm1 = LSTM(lstm_units, return_sequences=True)(attn_output)
            lstm1 = Dropout(dropout_rate)(lstm1)
            
            lstm2 = LSTM(lstm_units, return_sequences=False)(lstm1)
            x = Dropout(dropout_rate)(lstm2)
            
        elif self.attention_type == 'both':
            # Combination of Self-Attention and Standard Attention
            # Self-attention first
            self_attention = SelfAttentionLayer(d_model=lstm_units, num_heads=8)
            self_attn_output, _ = self_attention(inputs, inputs, inputs)
            # inputs projection to match self_attn_output shape
            inputs_proj = Dense(lstm_units)(inputs)
            self_attn_output = LayerNormalization()(self_attn_output + inputs_proj)
            
            # LSTM with return states
            lstm1 = LSTM(lstm_units, return_sequences=True, return_state=True)(self_attn_output)
            lstm_output, state_h, state_c = lstm1
            
            lstm2 = LSTM(lstm_units, return_sequences=True, return_state=True)(lstm_output)
            lstm_output2, state_h2, state_c2 = lstm2
            
            # Standard attention
            attention_layer = AttentionLayer(attention_units)
            context_vector, attention_weights = attention_layer(state_h2, lstm_output2)
            
            x = Dropout(dropout_rate)(context_vector)
        
        # Dense layers
        x = Dense(50, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(self.model.summary())
        
        return self.model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.1):
        """모델 학습"""
        print("모델을 학습하고 있습니다...")
        
        # 콜백 설정
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True,
            verbose=1
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=15, 
            min_lr=0.00001,
            verbose=1
        )
        
        # 모델 학습
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
    
    def get_attention_weights(self, X_sample):
        """어텐션 가중치 추출"""
        if self.attention_type in ['standard', 'both']:
            # Create a model that outputs attention weights
            attention_model = Model(
                inputs=self.model.input,
                outputs=[self.model.output] + [layer.output for layer in self.model.layers if 'attention' in layer.name.lower()]
            )
            
            try:
                outputs = attention_model.predict(X_sample[:1])
                if len(outputs) > 1:
                    # 항상 numpy array로 반환
                    return np.array(outputs[1])
            except:
                print("어텐션 가중치를 추출할 수 없습니다.")
        
        return None
    
    def evaluate_model(self):
        """모델 평가"""
        print("모델을 평가하고 있습니다...")
        
        # 예측
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # 어텐션 가중치 추출 (테스트 데이터 샘플)
        self.attention_weights = self.get_attention_weights(self.X_test)
        
        # 역정규화
        train_pred = self.scaler_y.inverse_transform(train_pred)
        test_pred = self.scaler_y.inverse_transform(test_pred)
        y_train_actual = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1))
        y_test_actual = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))
        
        # 성능 지표 계산
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        train_r2 = r2_score(y_train_actual, train_pred)
        test_r2 = r2_score(y_test_actual, test_pred)
        
        print(f"\n=== {self.target_currency.upper()} 환율 예측 성능 ({self.attention_type.title()} Attention) ===")
        print(f"학습 데이터 - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"테스트 데이터 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        return {
            'train_pred': train_pred.flatten(),
            'test_pred': test_pred.flatten(),
            'y_train_actual': y_train_actual.flatten(),
            'y_test_actual': y_test_actual.flatten(),
            'metrics': {
                'train_mae': train_mae, 'test_mae': test_mae,
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_r2': train_r2, 'test_r2': test_r2
            }
        }
    
    def plot_attention_weights(self):
        """어텐션 가중치 시각화"""
        if self.attention_weights is not None:
            # 항상 numpy array로 변환
            attention_weights = np.array(self.attention_weights)
            if len(attention_weights.shape) >= 2:
                plt.figure(figsize=(15, 8))
                
                # 첫 번째 샘플의 어텐션 가중치 시각화
                weights = attention_weights[0]
                if len(weights.shape) > 1:
                    weights = weights.squeeze()
                
                plt.subplot(2, 1, 1)
                plt.plot(weights)
                plt.title('Attention Weights Over Time Steps')
                plt.xlabel('Time Step')
                plt.ylabel('Attention Weight')
                plt.grid(True, alpha=0.3)
                
                # 히트맵으로 시각화 (여러 샘플)
                plt.subplot(2, 1, 2)
                sample_weights = attention_weights[:min(20, len(attention_weights))]
                if len(sample_weights.shape) > 2:
                    sample_weights = sample_weights.squeeze()
                
                if len(sample_weights.shape) == 2:
                    sns.heatmap(sample_weights, cmap='YlOrRd', cbar=True)
                    plt.title('Attention Weights Heatmap (20 samples)')
                    plt.xlabel('Time Step')
                    plt.ylabel('Sample')
                
                plt.tight_layout()
                plt.show()
            else:
                print("어텐션 가중치 시각화를 사용할 수 없습니다.")
        else:
            print("어텐션 가중치 시각화를 사용할 수 없습니다.")
    
    def plot_results(self, results):
        """결과 시각화"""
        plt.figure(figsize=(20, 12))
        
        # 1. 학습 손실 곡선
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss ({self.attention_type.title()} Attention)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 실제값 vs 예측값 (테스트 데이터)
        plt.subplot(2, 3, 2)
        plt.scatter(results['y_test_actual'], results['test_pred'], alpha=0.6)
        plt.plot([results['y_test_actual'].min(), results['y_test_actual'].max()], 
                [results['y_test_actual'].min(), results['y_test_actual'].max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.target_currency.upper()} - Actual vs Predicted (Test)')
        plt.grid(True, alpha=0.3)
        
        # 3. 잔차 플롯
        plt.subplot(2, 3, 3)
        residuals = results['y_test_actual'] - results['test_pred']
        plt.scatter(results['test_pred'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # 4. 시간별 예측 결과
        plt.subplot(2, 1, 2)
        dates = self.df.index[self.sequence_length:]
        train_size = len(results['y_train_actual'])
        
        plt.plot(dates[:train_size], results['y_train_actual'], label='Train Actual', alpha=0.8, linewidth=1.5)
        plt.plot(dates[:train_size], results['train_pred'], label='Train Predicted', alpha=0.8, linewidth=1.5)
        plt.plot(dates[train_size:], results['y_test_actual'], label='Test Actual', alpha=0.8, linewidth=2)
        plt.plot(dates[train_size:], results['test_pred'], label='Test Predicted', alpha=0.8, linewidth=2)
        
        plt.axvline(x=dates[train_size], color='r', linestyle='--', alpha=0.7, label='Train/Test Split')
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_currency.upper()} Exchange Rate')
        plt.title(f'{self.target_currency.upper()} Exchange Rate Prediction Over Time ({self.attention_type.title()} Attention)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 어텐션 가중치 시각화
        self.plot_attention_weights()
    
    def predict_future(self, days=30):
        """미래 환율 예측"""
        print(f"{days}일 후의 환율을 예측하고 있습니다...")
        
        # 최근 데이터로 예측
        last_sequence = self.scaler_X.transform(self.df[self.feature_columns].iloc[-self.sequence_length:].values)
        last_sequence = last_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for day in range(days):
            # 다음 값 예측
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # 시퀀스 업데이트 (더 정교한 방법)
            new_sequence = np.roll(current_sequence[0], -1, axis=0)
            
            # 예측값과 관련 피처들을 업데이트
            # 주요 통화 관련 피처들을 예측값으로 업데이트
            currency_indices = [i for i, col in enumerate(self.feature_columns) 
                              if any(curr in col for curr in ['usd', 'eur', 'gbp', 'jpy', 'cny'])]
            
            for idx in currency_indices[:min(5, len(currency_indices))]:
                new_sequence[-1, idx] = next_pred * (0.95 + np.random.random() * 0.1)  # 약간의 노이즈 추가
            
            current_sequence = new_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
        
        # 역정규화
        future_predictions = self.scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # 신뢰구간 계산 (단순한 방법)
        std_dev = np.std(future_predictions)
        upper_bound = future_predictions.flatten() + 1.96 * std_dev
        lower_bound = future_predictions.flatten() - 1.96 * std_dev
        
        # 미래 날짜 생성
        last_date = self.df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # 결과 시각화
        plt.figure(figsize=(15, 8))
        
        # 최근 60일 실제 데이터
        recent_data = self.df[self.target_currency].iloc[-90:]
        plt.plot(recent_data.index, recent_data.values, label='Historical Data', linewidth=2, color='blue')
        
        # 미래 예측
        plt.plot(future_dates, future_predictions.flatten(), 
                label=f'Future Prediction ({days} days)', linewidth=2, linestyle='--', color='red')
        
        # 신뢰구간
        plt.fill_between(future_dates, lower_bound, upper_bound, alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.axvline(x=last_date, color='green', linestyle=':', alpha=0.7, label='Today')
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_currency.upper()} Exchange Rate')
        plt.title(f'{self.target_currency.upper()} Exchange Rate - Future Prediction with {self.attention_type.title()} Attention')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_rate': future_predictions.flatten(),
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        })
    
    def run_full_pipeline(self, lstm_units=64, epochs=100, future_days=30):
        """전체 파이프라인 실행"""
        print("=" * 80)
        print(f"{self.target_currency.upper()} 환율 예측 Attention-LSTM 모델 실행 ({self.attention_type.title()} Attention)")
        print("=" * 80)
        
        # 1. 데이터 로드 및 전처리
        self.load_and_prepare_data()
        
        # 2. 피처 엔지니어링
        self.create_features()
        
        # 3. 학습/테스트 데이터 준비
        self.prepare_train_test_data()
        
        # 4. Attention 모델 구축
        self.build_attention_model(lstm_units=lstm_units)
        
        # 5. 모델 학습
        self.train_model(epochs=epochs)
        
        # 6. 모델 평가
        results = self.evaluate_model()
        
        # 7. 결과 시각화
        self.plot_results(results)
        
        # 8. 미래 예측
        future_pred = self.predict_future(days=future_days)
        
        print(f"\n미래 {future_days}일 예측 결과:")
        print(future_pred.head(10))
        
        return results, future_pred

# 모델 비교 함수
def compare_attention_models(target_currency='usd', sequence_length=60):
    """다양한 Attention 모델 비교"""
    print("=" * 80)
    print("Attention 메커니즘 비교")
    print("=" * 80)
    
    results_comparison = {}
    attention_types = ['standard', 'self', 'both']
    
    for attention_type in attention_types:
        print(f"\n{attention_type.title()} Attention 모델 실행 중...")
        
        predictor = ExchangeRateAttentionLSTMPredictor(
            target_currency=target_currency,
            sequence_length=sequence_length,
            attention_type=attention_type
        )
        
        try:
            results, future_pred = predictor.run_full_pipeline(lstm_units=64, epochs=50, future_days=30)
            results_comparison[attention_type] = results['metrics']
            
        except Exception as e:
            print(f"{attention_type} Attention 모델에서 오류 발생: {e}")
            results_comparison[attention_type] = None
    
    # 결과 비교 시각화
    if any(results_comparison.values()):
        plt.figure(figsize=(15, 10))
        
        metrics = ['test_mae', 'test_rmse', 'test_r2']
        metric_names = ['MAE', 'RMSE', 'R²']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            plt.subplot(2, 2, i+1)
            
            values = []
            labels = []
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            
            for j, (attention_type, result) in enumerate(results_comparison.items()):
                if result is not None:
                    values.append(result[metric])
                    labels.append(f'{attention_type.title()} Attention')
            
            bars = plt.bar(labels, values, color=colors[:len(labels)])
            plt.title(f'{name} 비교')
            plt.ylabel(name)
            plt.xticks(rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                        f'{value:.4f}', ha='center', va='bottom')
        
        # 종합 성능 표
        plt.subplot(2, 2, 4)
        plt.axis('tight')
        plt.axis('off')
        
        table_data = []
        headers = ['Attention Type', 'Test MAE', 'Test RMSE', 'Test R²']
        
        for attention_type, result in results_comparison.items():
            if result is not None:
                table_data.append([
                    f'{attention_type.title()} Attention',
                    f"{result['test_mae']:.4f}",
                    f"{result['test_rmse']:.4f}",
                    f"{result['test_r2']:.4f}"
                ])
        
        table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title('Performance Comparison Summary')
        
        plt.tight_layout()
        plt.show()
        
        # 최고 성능 모델 출력
        best_model = None
        best_r2 = -float('inf')
        
        for attention_type, result in results_comparison.items():
            if result is not None and result['test_r2'] > best_r2:
                best_r2 = result['test_r2']
                best_model = attention_type
        
        if best_model:
            print(f"\n최고 성능 모델: {best_model.title()} Attention (R² = {best_r2:.4f})")
    
    return results_comparison


# 사용 예시
if __name__ == "__main__":
    # 1. 단일 Attention 모델 실행
    print("=== Standard Attention LSTM 모델 ===")
    standard_predictor = ExchangeRateAttentionLSTMPredictor(
        target_currency='usd', 
        sequence_length=60, 
        attention_type='standard'
    )
    standard_results, standard_future = standard_predictor.run_full_pipeline(
        lstm_units=64, 
        epochs=50, 
        future_days=30
    )
    
    print("\n=== Self-Attention LSTM 모델 ===")
    self_predictor = ExchangeRateAttentionLSTMPredictor(
        target_currency='usd', 
        sequence_length=60, 
        attention_type='self'
    )
    self_results, self_future = self_predictor.run_full_pipeline(
        lstm_units=64, 
        epochs=50, 
        future_days=30
    )
    
    print("\n=== Both Attention LSTM 모델 ===")
    both_predictor = ExchangeRateAttentionLSTMPredictor(
        target_currency='usd', 
        sequence_length=60, 
        attention_type='both'
    )
    both_results, both_future = both_predictor.run_full_pipeline(
        lstm_units=64, 
        epochs=50, 
        future_days=30
    )
    
    # 2. 모델 비교 (선택사항)
    # comparison_results = compare_attention_models(target_currency='usd')
    
    # 3. 다른 통화에 대한 예측 (예시)
    """
    # EUR 예측
    eur_predictor = ExchangeRateAttentionLSTMPredictor(
        target_currency='eur', 
        attention_type='both'
    )
    eur_results, eur_future = eur_predictor.run_full_pipeline()
    
    # JPY 예측
    jpy_predictor = ExchangeRateAttentionLSTMPredictor(
        target_currency='jpy', 
        attention_type='self'
    )
    jpy_results, jpy_future = jpy_predictor.run_full_pipeline()
    """