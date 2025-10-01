import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        dilation_rate,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="causal",
        )
        self.norm1 = layers.LayerNormalization()
        self.relu1 = layers.Activation("relu")
        self.dropout1 = layers.Dropout(self.dropout_rate)

        self.conv2 = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="causal",
        )
        self.norm2 = layers.LayerNormalization()
        self.relu2 = layers.Activation("relu")
        self.dropout2 = layers.Dropout(self.dropout_rate)

        if input_shape[-1] != self.filters:
            self.residual_conv = layers.Conv1D(filters=self.filters, kernel_size=1)
        else:
            self.residual_conv = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)

        if self.residual_conv is not None:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs
        
        return layers.add([x, residual])


def build_model(look_back, num_features, tcn_filters=64, dropout_rate=0.2, learning_rate=0.001):
    inputs = keras.Input(shape=(look_back, num_features))

    x = inputs
    for i in range(3):  # 레이어 수 줄임
        dilation_rate = 2**i
        x = ResidualBlock(
            filters=tcn_filters,
            kernel_size=3,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
        )(x)

    # GlobalAveragePooling 대신 마지막 타임스텝 사용
    x = x[:, -1, :]  # 마지막 타임스텝만 사용
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0), 
        loss="mse",
        metrics=['mae']
    )
    
    return model
