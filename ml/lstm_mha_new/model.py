import tensorflow as tf

def build_model(
    look_back, num_features, lstm_units=150, dropout_rate=0.3, learning_rate=0.001
):
    """
    기존 Attention + Bidirectional LSTM 모델을 LSTM 모델로 변경합니다.
    """
    inputs = tf.keras.Input(shape=(look_back, num_features))
    
    # LSTM 레이어 (return_sequences=False가 기본값이지만 명시적으로 작성)
    # 마지막 Dense 레이어에 전체 시퀀스가 아닌 최종 출력만 전달합니다.
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
    
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="mse"
    )
    
    return model