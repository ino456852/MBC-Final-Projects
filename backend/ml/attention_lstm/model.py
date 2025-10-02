import tensorflow as tf


class CustomAttention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(input_shape[-1], activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1)
        super().build(input_shape)

    def call(self, x):
        a = tf.nn.softmax(self.dense2(self.dense1(x)), axis=1)
        return tf.reduce_sum(x * a, axis=1)


def build_model(seq_len: int, n_features: int, params: dict) -> tf.keras.Model:
    """LSTM + Attention 모델 생성"""
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
