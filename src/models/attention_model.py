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

# THIS DECORATOR IS THE FIX:
# registers BahdanauAttention with Keras so it can be
# correctly serialized when saving and deserialized
# when loading the .keras model file
@tf.keras.utils.register_keras_serializable(package='Custom')
class BahdanauAttention(Layer):
    """
    Soft attention over 49 spatial image regions.

    Inputs:
        features  : (batch, 49, 2048)  — spatial CNN features
        hidden    : (batch, 256)       — LSTM hidden state

    Returns:
        context   : (batch, 2048)      — weighted sum of regions
        weights   : (batch, 49)        — attention distribution (for visualization)
    """

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units

        # W1 projects image features: 2048 → units
        self.W1 = Dense(units)

        # W2 projects LSTM hidden state: 256 → units
        self.W2 = Dense(units)

        # V scores each location: units → 1
        self.V  = Dense(1)


    def call(self, features, hidden):

        # hidden: (batch, 256) → (batch, 1, 256)
        hidden_expanded = tf.expand_dims(hidden, 1)

        # score: (batch, 49, units)
        score = self.V(
            tf.nn.tanh(
                self.W1(features) + self.W2(hidden_expanded)
            )
        )

        # weights: (batch, 49, 1) → softmax over 49 locations
        attention_weights = tf.nn.softmax(score, axis=1)

        # context: weighted sum → (batch, 2048)
        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)

        # squeeze weights for output: (batch, 49)
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
    """
    Merge-architecture caption model with Bahdanau attention.

    Inputs:
        image_input  : (batch, 49, 2048)  — spatial features from ResNet50[-3]
        seq_input    : (batch, max_length) — tokenized partial caption

    Output:
        (batch, vocab_size) — next-word probability distribution
    """

    # -------------------------------------------------
    # IMAGE BRANCH
    # -------------------------------------------------

    # (batch, 49, 2048)
    image_input = Input(shape=(49, 2048), name='image_input')

    # light projection to reduce dimensionality before attention
    # (batch, 49, 256)
    image_features = Dense(
        256,
        activation='relu',
        name='image_projection'
    )(image_input)

    # dropout for regularization
    image_features = Dropout(0.4)(image_features)


    # -------------------------------------------------
    # SEQUENCE BRANCH
    # -------------------------------------------------

    # (batch, max_length)
    seq_input = Input(shape=(max_length,), name='seq_input')

    # (batch, max_length, 256)
    seq_embed = Embedding(
        vocab_size,
        256,
        mask_zero=True,
        name='embedding'
    )(seq_input)

    seq_embed = Dropout(0.4)(seq_embed)

    # LSTM returns both output sequence AND hidden state
    # lstm_out  : (batch, max_length, 256)
    # hidden    : (batch, 256)
    lstm_out, hidden_state, _ = LSTM(
        256,
        return_sequences=True,
        return_state=True,
        name='lstm'
    )(seq_embed)


    # -------------------------------------------------
    # ATTENTION
    # -------------------------------------------------

    attention_layer = BahdanauAttention(
        units=attention_units,
        name='bahdanau_attention'
    )

    # context : (batch, 256)  — attended image features
    # weights : (batch, 49)   — where the model is "looking"
    context, attention_weights = attention_layer(
        image_features,
        hidden_state
    )


    # -------------------------------------------------
    # DECODER
    # -------------------------------------------------

    # take last timestep output: (batch, 256)
    lstm_last = lstm_out[:, -1, :]

    # merge attended image context + LSTM output: (batch, 512)
    decoder_input = tf.concat([context, lstm_last], axis=-1)

    decoder = Dense(
        256,
        activation='relu',
        name='decoder_dense'
    )(decoder_input)

    decoder = Dropout(0.4)(decoder)

    # final word prediction: (batch, vocab_size)
    outputs = Dense(
        vocab_size,
        activation='softmax',
        name='output'
    )(decoder)


    # -------------------------------------------------
    # FINAL MODEL
    # -------------------------------------------------

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


# -------------------------------------------------
# QUICK TEST
# -------------------------------------------------

if __name__ == "__main__":

    VOCAB_SIZE  = 8768
    MAX_LENGTH  = 34

    model = build_caption_model(VOCAB_SIZE, MAX_LENGTH)

    model.summary()

    import numpy as np

    dummy_img = np.zeros((2, 49, 2048))
    dummy_seq = np.zeros((2, MAX_LENGTH))

    out = model.predict([dummy_img, dummy_seq], verbose=0)

    print("\nOutput shape:", out.shape)
    # expected: (2, 8768)

    print("Attention layer found:",
          any(isinstance(l, BahdanauAttention) for l in model.layers))