import sys
import os
import pickle
import argparse
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.models import Model

from src.config import BEST_MODEL_FILE, TOKENIZER_FILE
from src.models.attention_model import BahdanauAttention


# -------------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate caption using Attention-based Image Captioning model."
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_FILE,
        help="Path to trained attention model"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=TOKENIZER_FILE,
        help="Path to tokenizer"
    )

    parser.add_argument(
        "--beam_width",
        type=int,
        default=3,
        help="Beam width (1 = greedy decoding)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=34,
        help="Maximum caption length"
    )

    return parser.parse_args()


# -------------------------------------------------
# LOAD ATTENTION MODEL
# -------------------------------------------------

def load_caption_model(model_path):

    print(f"\nLoading model from: {model_path}")

    model = load_model(
        model_path,
        custom_objects={
            'BahdanauAttention': BahdanauAttention
        },
        compile=False
    )

    return model


# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

def load_tokenizer(tokenizer_path):

    print("Loading tokenizer...")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


# -------------------------------------------------
# LOAD RESNET50 FEATURE EXTRACTOR
# -------------------------------------------------

def load_feature_extractor():

    print("Loading ResNet50 feature extractor...")

    base_model = ResNet50(weights='imagenet')

    # spatial features: (7, 7, 2048)
    feature_extractor = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-3].output
    )

    return feature_extractor


# -------------------------------------------------
# EXTRACT IMAGE FEATURES
# -------------------------------------------------

def extract_features(image_path, feature_extractor):

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    # (1, 7, 7, 2048)
    feature = feature_extractor.predict(
        image,
        verbose=0
    )

    # reshape → (49, 2048)
    feature = feature.reshape(49, 2048)

    return feature


# -------------------------------------------------
# BUILD REVERSE WORD INDEX
# -------------------------------------------------

def build_index_to_word(tokenizer):

    return {
        index: word
        for word, index in tokenizer.word_index.items()
    }


# -------------------------------------------------
# GREEDY DECODING
# -------------------------------------------------

def greedy_decode(model,
                  image_feature,
                  tokenizer,
                  index_to_word,
                  max_length):

    caption = 'startseq'

    # (1, 49, 2048)
    image_input = np.expand_dims(
        image_feature,
        axis=0
    )

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences(
            [caption]
        )[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        )

        yhat = model.predict(
            [image_input, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = index_to_word.get(yhat)

        if word is None:
            break

        caption += ' ' + word

        if word == 'endseq':
            break

    return caption


# -------------------------------------------------
# BEAM SEARCH DECODING
# -------------------------------------------------

def beam_search_decode(model,
                       image_feature,
                       tokenizer,
                       index_to_word,
                       max_length,
                       beam_width=3):

    image_input = np.expand_dims(
        image_feature,
        axis=0
    )

    beams = [
        [['startseq'], 0.0]
    ]

    completed = []

    for _ in range(max_length):

        all_candidates = []

        for beam_tokens, beam_score in beams:

            if beam_tokens[-1] == 'endseq':

                completed.append(
                    [beam_tokens, beam_score]
                )

                continue

            caption_so_far = ' '.join(
                beam_tokens
            )

            sequence = tokenizer.texts_to_sequences(
                [caption_so_far]
            )[0]

            sequence = pad_sequences(
                [sequence],
                maxlen=max_length
            )

            preds = model.predict(
                [image_input, sequence],
                verbose=0
            )[0]

            top_indices = np.argsort(
                preds
            )[-beam_width:]

            for idx in top_indices:

                word = index_to_word.get(idx)

                if word is None:
                    continue

                candidate_score = (
                    beam_score +
                    np.log(preds[idx] + 1e-10)
                )

                all_candidates.append(
                    [
                        beam_tokens + [word],
                        candidate_score
                    ]
                )

        if not all_candidates:
            break

        all_candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = all_candidates[:beam_width]

    completed.extend(beams)

    completed.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_tokens = completed[0][0]

    return ' '.join(best_tokens)


# -------------------------------------------------
# CLEAN GENERATED CAPTION
# -------------------------------------------------

def clean_caption(caption):

    caption = caption.replace(
        'startseq',
        ''
    )

    caption = caption.replace(
        'endseq',
        ''
    )

    return caption.strip()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    args = parse_args()

    # validate image
    if not os.path.exists(args.image_path):

        print(
            f"Error: image not found -> {args.image_path}"
        )

        sys.exit(1)

    # load components
    caption_model = load_caption_model(
        args.model
    )

    tokenizer = load_tokenizer(
        args.tokenizer
    )

    feature_extractor = load_feature_extractor()

    index_to_word = build_index_to_word(
        tokenizer
    )

    # extract image features
    print(f"\nProcessing image: {args.image_path}")

    image_feature = extract_features(
        args.image_path,
        feature_extractor
    )

    # generate caption
    if args.beam_width == 1:

        print("\nUsing greedy decoding...")

        raw_caption = greedy_decode(
            caption_model,
            image_feature,
            tokenizer,
            index_to_word,
            args.max_length
        )

    else:

        print(
            f"\nUsing beam search "
            f"(beam width = {args.beam_width})..."
        )

        raw_caption = beam_search_decode(
            caption_model,
            image_feature,
            tokenizer,
            index_to_word,
            args.max_length,
            beam_width=args.beam_width
        )

    final_caption = clean_caption(
        raw_caption
    )

    # display result
    print("\n" + "=" * 60)

    print("Generated Caption:\n")

    print(final_caption)

    print("=" * 60)


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    main()