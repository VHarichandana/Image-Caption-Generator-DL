import os
import pickle
import string
import numpy as np
import pandas as pd

from tqdm import tqdm

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from src.config import (
    CAPTIONS_FILE,
    IMAGE_FEATURES_FILE,
    SAVED_MODELS_DIR,
    TOKENIZER_FILE,
    CAPTION_MODEL_FILE
)

from src.models.caption_model import build_caption_model


# -------------------------------------------------
# LOAD FEATURES
# -------------------------------------------------

with open(IMAGE_FEATURES_FILE, "rb") as f:
    features = pickle.load(f)


# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

with open(TOKENIZER_FILE, "rb") as f:
    tokenizer = pickle.load(f)


# -------------------------------------------------
# VARIABLES
# -------------------------------------------------

vocab_size = len(tokenizer.word_index) + 1

max_length = 34


# -------------------------------------------------
# CLEAN CAPTION
# -------------------------------------------------

def clean_caption(caption):

    caption = caption.lower()

    caption = caption.translate(
        str.maketrans('', '', string.punctuation)
    )

    caption = ''.join(
        char for char in caption
        if char.isalpha() or char == ' '
    )

    caption = caption.split()

    caption = [
        word for word in caption
        if len(word) > 1
    ]

    caption = ' '.join(caption)

    return caption


# -------------------------------------------------
# LOAD CLEAN MAPPING
# -------------------------------------------------

def load_clean_mapping():

    df = pd.read_csv(CAPTIONS_FILE)

    mapping = {}

    for _, row in df.iterrows():

        image_name = row['image']
        caption = row['caption']

        caption = clean_caption(caption)

        caption = 'startseq ' + caption + ' endseq'

        if image_name not in mapping:
            mapping[image_name] = []

        mapping[image_name].append(caption)

    return mapping


mapping = load_clean_mapping()


# -------------------------------------------------
# DATA GENERATOR
# -------------------------------------------------

def data_generator(mapping,
                   features,
                   tokenizer,
                   max_length,
                   vocab_size,
                   batch_size):

    X1, X2, y = [], [], []

    n = 0

    while True:

        for image_name, captions in mapping.items():

            n += 1

            image_feature = features[image_name]

            for caption in captions:

                sequence = tokenizer.texts_to_sequences(
                    [caption]
                )[0]

                for i in range(1, len(sequence)):

                    in_seq = sequence[:i]

                    out_seq = sequence[i]

                    in_seq = pad_sequences(
                        [in_seq],
                        maxlen=max_length
                    )[0]

                    out_seq = to_categorical(
                        [out_seq],
                        num_classes=vocab_size
                    )[0]

                    X1.append(image_feature)

                    X2.append(in_seq)

                    y.append(out_seq)

            if n == batch_size:

                yield (
                    (
                        np.array(X1),
                        np.array(X2)
                    ),
                    np.array(y)
                )

                X1, X2, y = [], [], []

                n = 0


# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------

model = build_caption_model(
    vocab_size,
    max_length
)

print(model.summary())


# -------------------------------------------------
# TRAINING
# -------------------------------------------------

epochs = 10

batch_size = 32

steps = len(mapping) // batch_size

generator = data_generator(
    mapping,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size
)

print("\nStarting Training...\n")

model.fit(
    generator,
    epochs=epochs,
    steps_per_epoch=steps,
    verbose=1
)


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

model.save(
    CAPTION_MODEL_FILE
)

print("\nModel saved successfully.")