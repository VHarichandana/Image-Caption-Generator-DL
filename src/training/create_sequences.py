import os
import pickle
import string
import numpy as np
import pandas as pd

from src.config import CAPTIONS_FILE, IMAGE_FEATURES_FILE, TOKENIZER_FILE

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# -------------------------------------------------
# LOAD FEATURES
# -------------------------------------------------

def load_features():

    with open(IMAGE_FEATURES_FILE, "rb") as f:
        features = pickle.load(f)

    return features


# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

def load_tokenizer():

    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


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


# -------------------------------------------------
# CREATE TRAINING SEQUENCES
# -------------------------------------------------

def create_sequences(tokenizer,
                     max_length,
                     captions_list,
                     image_feature,
                     vocab_size):

    X1, X2, y = [], [], []

    for caption in captions_list:

        # caption -> integer sequence
        sequence = tokenizer.texts_to_sequences(
            [caption]
        )[0]

        # create multiple samples
        for i in range(1, len(sequence)):

            # input sequence
            in_seq = sequence[:i]

            # target word
            out_seq = sequence[i]

            # pad input
            in_seq = pad_sequences(
                [in_seq],
                maxlen=max_length
            )[0]

            # one-hot encode target
            out_seq = to_categorical(
                [out_seq],
                num_classes=vocab_size
            )[0]

            # store
            X1.append(image_feature)
            X2.append(in_seq)
            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    print("Loading image features...")

    features = load_features()

    print("Loading tokenizer...")

    tokenizer = load_tokenizer()

    print("Loading cleaned captions...")

    mapping = load_clean_mapping()

    vocab_size = len(tokenizer.word_index) + 1

    max_length = 34

    print("\nVocabulary Size:", vocab_size)
    print("Maximum Caption Length:", max_length)

    # test on one image
    first_image = list(mapping.keys())[0]

    print("\nSample Image:")
    print(first_image)

    captions = mapping[first_image]

    image_feature = features[first_image]

    print("\nGenerating sequences...")

    X1, X2, y = create_sequences(
        tokenizer,
        max_length,
        captions,
        image_feature,
        vocab_size
    )

    print("\nShapes:")
    print("X1 shape:", X1.shape)
    print("X2 shape:", X2.shape)
    print("y shape :", y.shape)

    print("\nSequence generation completed successfully.")


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    main()