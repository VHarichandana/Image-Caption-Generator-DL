import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Layer
)


# -------------------------------------------------
# BAHDANAU ATTENTION LAYER
# -------------------------------------------------

@tf.keras.utils.register_keras_serializable(package='Custom')
class BahdanauAttention(Layer):

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)

    def call(self, features, hidden):

        hidden_expanded = tf.expand_dims(hidden, 1)

        score = self.V(
            tf.nn.tanh(
                self.W1(features) + self.W2(hidden_expanded)
            )
        )

        attention_weights = tf.nn.softmax(score, axis=1)

        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)

        attention_weights = tf.squeeze(attention_weights, axis=-1)

        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


# -------------------------------------------------
# BUILD CAPTION MODEL
# -------------------------------------------------

def build_caption_model(vocab_size, max_length, attention_units=256):

    image_input = Input(shape=(49, 2048), name='image_input')

    image_features = Dense(
        256,
        activation='relu',
        name='image_projection'
    )(image_input)

    image_features = Dropout(0.4)(image_features)

    seq_input = Input(shape=(max_length,), name='seq_input')

    seq_embed = Embedding(
        vocab_size,
        256,
        mask_zero=True,
        name='embedding'
    )(seq_input)

    seq_embed = Dropout(0.4)(seq_embed)

    lstm_out, hidden_state, _ = LSTM(
        256,
        return_sequences=True,
        return_state=True,
        name='lstm'
    )(seq_embed)

    attention_layer = BahdanauAttention(
        units=attention_units,
        name='bahdanau_attention'
    )

    context, _ = attention_layer(image_features, hidden_state)

    lstm_last = lstm_out[:, -1, :]

    decoder_input = tf.concat([context, lstm_last], axis=-1)

    decoder = Dense(
        256,
        activation='relu',
        name='decoder_dense'
    )(decoder_input)

    decoder = Dropout(0.4)(decoder)

    outputs = Dense(
        vocab_size,
        activation='softmax',
        name='output'
    )(decoder)

    model = Model(
        inputs=[image_input, seq_input],
        outputs=outputs,
        name='attention_caption_model'
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model