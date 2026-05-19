import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from PIL import Image

from src.models.attention_model import BahdanauAttention


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Caption Generator")

st.write("Upload an image and let AI generate a caption.")


# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------

VOCAB_SIZE     = 8768
MAX_LENGTH     = 34
MODEL_PATH     = "saved_models/caption_model.keras"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_caption_model():

    # custom_object_scope is the most reliable way to inject
    # a custom layer class during model loading — works even
    # if the model was saved before the decorator was added
    with tf.keras.utils.custom_object_scope(
        {'BahdanauAttention': BahdanauAttention}
    ):
        model = load_model(MODEL_PATH)

    return model


caption_model = load_caption_model()


# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

@st.cache_resource
def load_tokenizer():

    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


tokenizer = load_tokenizer()


# -------------------------------------------------
# FEATURE EXTRACTOR
# -------------------------------------------------

@st.cache_resource
def load_feature_extractor():

    base_model = ResNet50(weights='imagenet')

    return Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-3].output
    )


feature_extractor = load_feature_extractor()


# -------------------------------------------------
# EXTRACT FEATURES
# -------------------------------------------------

def extract_features(image):

    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # (1, 7, 7, 2048) → (1, 49, 2048)
    feature = feature_extractor.predict(image, verbose=0)
    feature = feature.reshape((1, 49, 2048))

    return feature


# -------------------------------------------------
# INTEGER -> WORD
# -------------------------------------------------

def build_index_to_word(tokenizer):
    """Build reverse lookup dict once — O(1) per word."""
    return {index: word for word, index in tokenizer.word_index.items()}


# -------------------------------------------------
# GREEDY DECODE
# -------------------------------------------------

def greedy_decode(model, image_feature, tokenizer,
                  index_to_word, max_length):

    caption = 'startseq'

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = index_to_word.get(yhat)
        if word is None:
            break

        caption += ' ' + word
        if word == 'endseq':
            break

    return caption


# -------------------------------------------------
# BEAM SEARCH DECODE
# -------------------------------------------------

def beam_search_decode(model, image_feature, tokenizer,
                       index_to_word, max_length, beam_width=3):

    beams     = [[['startseq'], 0.0]]
    completed = []

    for _ in range(max_length):

        all_candidates = []

        for beam_tokens, beam_score in beams:

            if beam_tokens[-1] == 'endseq':
                completed.append([beam_tokens, beam_score])
                continue

            caption_so_far = ' '.join(beam_tokens)
            sequence = tokenizer.texts_to_sequences([caption_so_far])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)

            preds = model.predict(
                [image_feature, sequence],
                verbose=0
            )[0]

            top_indices = np.argsort(preds)[-beam_width:]

            for idx in top_indices:

                word = index_to_word.get(idx)
                if word is None:
                    continue

                score = beam_score + np.log(preds[idx] + 1e-10)
                all_candidates.append([beam_tokens + [word], score])

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    completed.extend(beams)
    completed.sort(key=lambda x: x[1], reverse=True)

    return ' '.join(completed[0][0])


# -------------------------------------------------
# CLEAN CAPTION
# -------------------------------------------------

def clean_caption(caption):
    caption = caption.replace('startseq', '')
    caption = caption.replace('endseq', '')
    return caption.strip()


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

index_to_word = build_index_to_word(tokenizer)

beam_width = st.slider(
    "Beam width",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    help="1 = greedy decoding · 2–6 = beam search (higher = better quality, slower)"
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Generating caption..."):

        image_feature = extract_features(image)

        if beam_width == 1:
            raw_caption = greedy_decode(
                caption_model,
                image_feature,
                tokenizer,
                index_to_word,
                MAX_LENGTH
            )
        else:
            raw_caption = beam_search_decode(
                caption_model,
                image_feature,
                tokenizer,
                index_to_word,
                MAX_LENGTH,
                beam_width=beam_width
            )

        final_caption = clean_caption(raw_caption)

    st.success("Caption generated!")
    st.subheader("Generated Caption:")
    st.write(final_caption)