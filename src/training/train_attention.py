import os
import pickle
import string
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from src.config import (
    CAPTIONS_FILE,
    SAVED_MODELS_DIR,
    ATTENTION_FEATURES_FILE,
    TOKENIZER_FILE,
    BEST_MODEL_FILE,
    ATTENTION_MODEL_FILE
)

from src.models.caption_model import build_caption_model


# -------------------------------------------------
# LOAD ATTENTION FEATURES
# -------------------------------------------------

print("Loading attention features...")

with open(ATTENTION_FEATURES_FILE, "rb") as f:
    features = pickle.load(f)

# each value shape: (49, 2048)
print(f"Features loaded: {len(features)} images")


# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

print("Loading tokenizer...")

with open(TOKENIZER_FILE, "rb") as f:
    tokenizer = pickle.load(f)


# -------------------------------------------------
# VARIABLES
# -------------------------------------------------

vocab_size  = len(tokenizer.word_index) + 1
max_length  = 34
batch_size  = 32
epochs      = 5

print(f"Vocab size : {vocab_size}")
print(f"Max length : {max_length}")


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
        caption    = row['caption']

        caption = clean_caption(caption)
        caption = 'startseq ' + caption + ' endseq'

        if image_name not in mapping:
            mapping[image_name] = []

        mapping[image_name].append(caption)

    return mapping


print("Loading captions...")

mapping = load_clean_mapping()

print(f"Total images with captions: {len(mapping)}")


# -------------------------------------------------
# TRAIN / VAL SPLIT  (80 / 20)
# -------------------------------------------------

all_image_names = list(mapping.keys())

train_names, val_names = train_test_split(
    all_image_names,
    test_size=0.20,
    random_state=42
)

train_mapping = {k: mapping[k] for k in train_names}
val_mapping   = {k: mapping[k] for k in val_names}

print(f"\nTrain images : {len(train_mapping)}")
print(f"Val   images : {len(val_mapping)}")


# -------------------------------------------------
# DATA GENERATOR
# -------------------------------------------------

def data_generator(mapping,
                   features,
                   tokenizer,
                   max_length,
                   vocab_size,
                   batch_size):
    """
    Yields batches of:
        X1 : image spatial features  (batch, 49, 2048)
        X2 : partial caption sequence (batch, max_length)
        y  : one-hot next word        (batch, vocab_size)
    """

    X1, X2, y = [], [], []

    while True:

        for image_name, captions in mapping.items():

            if image_name not in features:
                continue

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

                    if len(y) == batch_size:
                        yield (
                            (
                                np.array(X1),   # (batch, 49, 2048)
                                np.array(X2)    # (batch, max_length)
                            ),
                            np.array(y)         # (batch, vocab_size)
                        )

                        X1, X2, y = [], [], []


def count_samples(mapping, tokenizer):
    total = 0

    for image_name, captions in mapping.items():

        if image_name not in features:
            continue

        for caption in captions:
            sequence = tokenizer.texts_to_sequences([caption])[0]
            total += max(0, len(sequence) - 1)

    return total


# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------

print("\nBuilding model...")

model = build_caption_model(vocab_size, max_length)

model.summary()


# -------------------------------------------------
# CALLBACKS
# -------------------------------------------------

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

callbacks = [

    # save best model based on val_loss
    ModelCheckpoint(
        filepath=BEST_MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # stop early if val_loss stops improving
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # reduce LR when val_loss plateaus
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# -------------------------------------------------
# STEPS PER EPOCH
# -------------------------------------------------

train_steps = 500
val_steps = 100

print(f"\nTrain steps/epoch : {train_steps}")
print(f"Val   steps/epoch : {val_steps}")


# -------------------------------------------------
# GENERATORS
# -------------------------------------------------

train_generator = data_generator(
    train_mapping,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size
)

val_generator = data_generator(
    val_mapping,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size
)


# -------------------------------------------------
# TRAIN
# -------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)


# -------------------------------------------------
# SAVE FINAL MODEL
# -------------------------------------------------

model.save(ATTENTION_MODEL_FILE)

print(f"\nFinal model saved to {ATTENTION_MODEL_FILE}")
print(f"Best model saved to {BEST_MODEL_FILE}")


# -------------------------------------------------
# TRAINING SUMMARY
# -------------------------------------------------

best_epoch = np.argmin(history.history['val_loss']) + 1
best_val   = min(history.history['val_loss'])
final_acc  = history.history['accuracy'][-1]

print(f"\nTraining complete.")
print(f"Best epoch     : {best_epoch}")
print(f"Best val loss  : {best_val:.4f}")
print(f"Final train acc: {final_acc:.4f}")