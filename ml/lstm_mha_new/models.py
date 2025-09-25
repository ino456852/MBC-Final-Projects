import tensorflow as tf


def get_optimizer(name, learning_rate):
    if name == "AdamW":
        return tf.keras.optimizers.AdamW(learning_rate)
    if name == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate)
    if name == "Adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=1.0)
    if name == "Adafactor":
        return tf.keras.optimizers.Adafactor()
    return tf.keras.optimizers.Adam(learning_rate)


def create_lstm_encoder(look_back, num_features, params):
    inputs = tf.keras.Input(shape=(look_back, num_features))
    lstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(params["lstm_units"], return_sequences=True)
    )(inputs)
    return inputs, lstm_out


def finalize_model(inputs, attention_output, params):
    dropout_out = tf.keras.layers.Dropout(params["dropout_rate"])(attention_output)
    outputs = tf.keras.layers.Dense(1)(dropout_out)
    model = tf.keras.Model(inputs, outputs)

    optimizer = get_optimizer(
        getattr(params, "optimizer_name", "Adam"), params["learning_rate"]
    )
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


class SumOverTime(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1, self.dense2 = None, None

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(input_shape[-1], activation="tanh")
        self.dense2 = tf.keras.layers.Dense(1)
        super().build(input_shape)

    def call(self, x):
        uit = self.dense1(x)
        ait = self.dense2(uit)
        a = tf.nn.softmax(ait, axis=1)
        return tf.reduce_sum(x * a, axis=1)


def build_custom_attention_model(look_back, num_features, params):
    inputs, lstm_out = create_lstm_encoder(look_back, num_features, params)
    attention_out = CustomAttention()(lstm_out)
    return finalize_model(inputs, attention_out, params)


def build_multi_head_attention_model(look_back, num_features, params):
    inputs, lstm_out = create_lstm_encoder(look_back, num_features, params)
    attention_out = tf.keras.layers.MultiHeadAttention(
        num_heads=params["num_heads"], key_dim=params["key_dim"]
    )(query=lstm_out, value=lstm_out, key=lstm_out)
    pooled_out = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
    return finalize_model(inputs, pooled_out, params)


def build_dot_product_attention_model(look_back, num_features, params):
    inputs, lstm_out = create_lstm_encoder(look_back, num_features, params)
    attention_weights = tf.keras.layers.Dense(1, activation="tanh")(lstm_out)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
    attention_context = tf.keras.layers.multiply([lstm_out, attention_weights])
    attention_out = SumOverTime()(attention_context)
    return finalize_model(inputs, attention_out, params)


def build_self_attention_model(look_back, num_features, params):
    inputs, lstm_out = create_lstm_encoder(look_back, num_features, params)
    attention_out = tf.keras.layers.Attention()([lstm_out, lstm_out])
    pooled_out = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
    return finalize_model(inputs, pooled_out, params)


def build_causal_self_attention_model(look_back, num_features, params):
    inputs, lstm_out = create_lstm_encoder(look_back, num_features, params)
    mha = tf.keras.layers.MultiHeadAttention(
        num_heads=params["num_heads"], key_dim=params["key_dim"]
    )
    attention_out = mha(
        query=lstm_out, value=lstm_out, key=lstm_out, use_causal_mask=True
    )
    pooled_out = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
    return finalize_model(inputs, pooled_out, params)


def build_cross_attention_model(look_back, num_base_features, num_features, params):
    inputs = tf.keras.Input(shape=(look_back, num_features))

    base_features_input = inputs[:, :, :num_base_features]
    target_related_input = inputs[:, :, num_base_features:]

    base_lstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(params["lstm_units"], return_sequences=True)
    )(base_features_input)
    target_lstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(params["lstm_units_half"], return_sequences=True)
    )(target_related_input)

    attention_out = tf.keras.layers.MultiHeadAttention(
        num_heads=params["num_heads"], key_dim=params["key_dim"]
    )(query=target_lstm_out, value=base_lstm_out, key=base_lstm_out)

    pooled_out = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
    return finalize_model(inputs, pooled_out, params)


# 다른 모델의 성능이 좋지 않아서 mha 모델만 학습하기로 함
MODELS_TO_TRAIN = {
    # "custom": build_custom_attention_model,
    "mha": build_multi_head_attention_model,
    # "dot": build_dot_product_attention_model,
    # "cross": build_cross_attention_model,
    # "causal": build_causal_self_attention_model,
    # "self": build_self_attention_model,
}
