import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.config import CAPTIONS_FILE, SAVED_MODELS_DIR, TOKENIZER_FILE

import string
import pickle
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer


# -----------------------------
# LOAD CAPTIONS
# -----------------------------

def load_captions(captions_path):

    df = pd.read_csv(captions_path)

    return df


# -----------------------------
# CREATE IMAGE -> CAPTIONS MAPPING
# -----------------------------

def create_mapping(df):

    mapping = {}

    for _, row in df.iterrows():

        image_name = row['image']
        caption = row['caption']

        if image_name not in mapping:
            mapping[image_name] = []

        mapping[image_name].append(caption)

    return mapping


# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_caption(caption):

    # lowercase
    caption = caption.lower()

    # remove punctuation
    caption = caption.translate(
        str.maketrans('', '', string.punctuation)
    )

    # remove numbers
    caption = ''.join(
        char for char in caption
        if char.isalpha() or char == ' '
    )

    # tokenize
    caption = caption.split()

    # remove short words
    caption = [
        word for word in caption
        if len(word) > 1
    ]

    # join back
    caption = ' '.join(caption)

    return caption


# -----------------------------
# CLEAN ALL CAPTIONS
# -----------------------------

def clean_all_captions(mapping):

    for image_name, captions in mapping.items():

        cleaned_captions = []

        for caption in captions:

            cleaned = clean_caption(caption)

            # add start/end tokens
            cleaned = 'startseq ' + cleaned + ' endseq'

            cleaned_captions.append(cleaned)

        mapping[image_name] = cleaned_captions

    return mapping


# -----------------------------
# GET ALL CAPTIONS
# -----------------------------

def get_all_captions(mapping):

    all_captions = []

    for captions in mapping.values():
        all_captions.extend(captions)

    return all_captions


# -----------------------------
# CREATE TOKENIZER
# -----------------------------

def create_tokenizer(all_captions):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(all_captions)

    return tokenizer


# -----------------------------
# FIND MAX LENGTH
# -----------------------------

def get_max_length(all_captions):

    return max(
        len(caption.split())
        for caption in all_captions
    )


# -----------------------------
# SAVE TOKENIZER
# -----------------------------

def save_tokenizer(tokenizer):

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)


# -----------------------------
# MAIN FUNCTION
# -----------------------------

def main():

    captions_path = CAPTIONS_FILE

    print("Loading captions...")

    df = load_captions(captions_path)

    print(df.head())

    print("\nCreating mapping...")

    mapping = create_mapping(df)

    first_key = list(mapping.keys())[0]

    print("\nSample Image:")
    print(first_key)

    print("\nSample Captions:")
    print(mapping[first_key])

    print("\nCleaning captions...")

    mapping = clean_all_captions(mapping)

    print("\nCleaned Caption:")
    print(mapping[first_key][0])

    print("\nCollecting all captions...")

    all_captions = get_all_captions(mapping)

    print("Total Captions:", len(all_captions))

    print("\nCreating tokenizer...")

    tokenizer = create_tokenizer(all_captions)

    vocab_size = len(tokenizer.word_index) + 1

    print("Vocabulary Size:", vocab_size)

    print("\nExample Word Index:")
    print("dog ->", tokenizer.word_index.get("dog"))

    sample_caption = "startseq dog running endseq"

    sequence = tokenizer.texts_to_sequences(
        [sample_caption]
    )[0]

    print("\nSample Sequence:")
    print(sequence)

    max_length = get_max_length(all_captions)

    print("\nMaximum Caption Length:", max_length)

    print("\nSaving tokenizer...")

    save_tokenizer(tokenizer)

    print("\nTokenizer saved successfully!")

    print("\nPreprocessing Completed Successfully.")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    main()