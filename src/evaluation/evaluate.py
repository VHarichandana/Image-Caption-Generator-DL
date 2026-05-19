import os
import pickle
import string
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from nltk.translate.bleu_score import corpus_bleu

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from src.config import (
    CAPTIONS_FILE,
    IMAGES_DIR,
    ATTENTION_FEATURES_FILE,
    TOKENIZER_FILE,
    BEST_MODEL_FILE
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.models import Model

from src.models.attention_model import BahdanauAttention
# ARGUMENT PARSER

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate Image Caption Generator"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_FILE,
        help="Path to trained attention model"
    )

    parser.add_argument(
        "--features",
        type=str,
        default=ATTENTION_FEATURES_FILE,
        help="Path to extracted attention features"
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
        help="Beam search width"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=34,
        help="Maximum caption length"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate"
    )

    parser.add_argument(
        "--show_samples",
        type=int,
        default=5,
        help="Number of sample predictions to display"
    )

    return parser.parse_args()

# CLEAN CAPTION

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
# LOAD CLEAN MAPPING

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

# TEST SPLIT

def get_test_mapping(mapping):

    all_image_names = list(mapping.keys())

    _, test_names = train_test_split(
        all_image_names,
        test_size=0.20,
        random_state=42
    )

    test_mapping = {
        k: mapping[k]
        for k in test_names
    }

    return test_mapping

# LOAD FEATURE EXTRACTOR

def load_feature_extractor():

    base_model = ResNet50(weights='imagenet')

    feature_extractor = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-3].output
    )

    return feature_extractor
# EXTRACT IMAGE FEATURES

def extract_features(image_path, feature_extractor):

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = feature_extractor.predict(
        image,
        verbose=0
    )

    feature = feature.reshape(49, 2048)

    return feature

# GET FEATURE

def get_image_feature(image_name,
                      features_cache,
                      feature_extractor):

    if image_name in features_cache:
        return features_cache[image_name]

    image_path = os.path.join(
        IMAGES_DIR,
        image_name
    )

    feature = extract_features(
        image_path,
        feature_extractor
    )

    return feature

# REVERSE TOKENIZER

def build_index_to_word(tokenizer):

    return {
        index: word
        for word, index
        in tokenizer.word_index.items()
    }

# GREEDY DECODING

def greedy_decode(model,
                  image_feature,
                  tokenizer,
                  index_to_word,
                  max_length):

    caption = 'startseq'

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

# BEAM SEARCH DECODING


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

                score = (
                    beam_score
                    + np.log(preds[idx] + 1e-10)
                )

                all_candidates.append(
                    [beam_tokens + [word], score]
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

# REMOVE SPECIAL TOKENS

def strip_tokens(caption):

    caption = caption.replace(
        'startseq',
        ''
    )

    caption = caption.replace(
        'endseq',
        ''
    )

    return caption.strip()

# MAIN

def main():

    args = parse_args()
    # LOAD MODEL
    print("Loading caption model...")

    model = load_model(
        args.model,
        custom_objects={
            'BahdanauAttention': BahdanauAttention
        },
        compile=False
    )

    # LOAD TOKENIZER


    print("Loading tokenizer...")

    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    # LOAD FEATURES

    print("Loading attention features...")

    with open(args.features, "rb") as f:
        features_cache = pickle.load(f)

    # LOAD MAPPING

    print("Loading captions...")

    mapping = load_clean_mapping()

    test_mapping = get_test_mapping(mapping)

    print(f"Total Images : {len(mapping)}")
    print(f"Test Images  : {len(test_mapping)}")

    test_items = list(test_mapping.items())

    if args.num_samples is not None:

        test_items = test_items[:args.num_samples]

        print(
            f"Evaluating on {args.num_samples} samples"
        )
    # LOAD FEATURE EXTRACTOR

    feature_extractor = load_feature_extractor()

    index_to_word = build_index_to_word(
        tokenizer
    )

    # EVALUATION
    actual = []
    predicted = []

    print("\nGenerating captions...\n")

    for image_name, captions in tqdm(test_items):

        image_feature = get_image_feature(
            image_name,
            features_cache,
            feature_extractor
        )

        if args.beam_width == 1:

            generated_caption = greedy_decode(
                model,
                image_feature,
                tokenizer,
                index_to_word,
                args.max_length
            )

        else:

            generated_caption = beam_search_decode(
                model,
                image_feature,
                tokenizer,
                index_to_word,
                args.max_length,
                beam_width=args.beam_width
            )

        predicted_caption = strip_tokens(
            generated_caption
        ).split()

        reference_captions = [
            strip_tokens(cap).split()
            for cap in captions
        ]

        actual.append(reference_captions)

        predicted.append(predicted_caption)
    # BLEU SCORES
    bleu1 = corpus_bleu(
        actual,
        predicted,
        weights=(1.0, 0, 0, 0)
    )

    bleu2 = corpus_bleu(
        actual,
        predicted,
        weights=(0.5, 0.5, 0, 0)
    )

    bleu3 = corpus_bleu(
        actual,
        predicted,
        weights=(0.33, 0.33, 0.33, 0)
    )

    bleu4 = corpus_bleu(
        actual,
        predicted,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

    print("\n" + "=" * 50)

    print("BLEU SCORES")

    print("=" * 50)

    print(f"BLEU-1 : {bleu1:.4f}")
    print(f"BLEU-2 : {bleu2:.4f}")
    print(f"BLEU-3 : {bleu3:.4f}")
    print(f"BLEU-4 : {bleu4:.4f}")

    print("=" * 50)

    # SAMPLE PREDICTIONS


    print(
        f"\nSample Predictions "
        f"(first {args.show_samples}):\n"
    )

    for image_name, captions in test_items[:args.show_samples]:

        image_feature = get_image_feature(
            image_name,
            features_cache,
            feature_extractor
        )

        if args.beam_width == 1:

            generated_caption = greedy_decode(
                model,
                image_feature,
                tokenizer,
                index_to_word,
                args.max_length
            )

        else:

            generated_caption = beam_search_decode(
                model,
                image_feature,
                tokenizer,
                index_to_word,
                args.max_length,
                beam_width=args.beam_width
            )

        final_caption = strip_tokens(
            generated_caption
        )

        print(f"\nImage : {image_name}")

        print(f"Generated : {final_caption}")

        print("\nGround Truth:")

        for cap in captions:

            print(
                "-",
                strip_tokens(cap)
            )

        print("\n" + "-" * 50)
# ENTRY POINT

if __name__ == "__main__":
    main()