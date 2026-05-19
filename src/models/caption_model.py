import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Concatenate
)

from src.models.attention_model import BahdanauAttention


def build_caption_model(vocab_size, max_length, attention_units=256):
    """Build the attention-based caption model."""

    image_input = Input(
        shape=(49, 2048),
        name='image_input'
    )

    image_features = Dense(
        256,
        activation='relu',
        name='image_projection'
    )(image_input)

    image_features = Dropout(0.4)(image_features)

    seq_input = Input(
        shape=(max_length,),
        name='seq_input'
    )

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

    context, _ = BahdanauAttention(
        units=attention_units,
        name='bahdanau_attention'
    )(image_features, hidden_state)

    lstm_last = lstm_out[:, -1, :]

    decoder_input = Concatenate(axis=-1)([context, lstm_last])

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
