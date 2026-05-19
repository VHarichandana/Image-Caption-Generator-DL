import streamlit as st
import pickle
import numpy as np

from src.models.caption_model import BahdanauAttention
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from src.config import BEST_MODEL_FILE, TOKENIZER_FILE
from src.models.attention_model import BahdanauAttention
from PIL import Image


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Caption Generator")

st.write(
    "Upload an image and let AI generate a caption."
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_caption_model():
    model = load_model(
        "saved_models/caption_model.keras",
        custom_objects={'BahdanauAttention': BahdanauAttention}
    )

caption_model = load_caption_model()


# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_FILE, "rb") as f:
        return pickle.load(f)


tokenizer = load_tokenizer()


# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------

max_length = 34


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

    feature = feature_extractor.predict(image, verbose=0)
    feature = feature.reshape((1, 49, 2048))

    return feature


# -------------------------------------------------
# INTEGER -> WORD
# -------------------------------------------------

def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# -------------------------------------------------
# GREEDY DECODE
# -------------------------------------------------

def greedy_decode(model, image_feature, tokenizer, max_length):
    caption = 'startseq'

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = index_to_word(yhat, tokenizer)
        if word is None:
            break

        caption += ' ' + word
        if word == 'endseq':
            break

    return caption


# -------------------------------------------------
# BEAM SEARCH DECODE
# -------------------------------------------------

def beam_search_decode(model, image_feature, tokenizer, max_length, beam_width=3):
    beams = [[['startseq'], 0.0]]
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

            preds = model.predict([image_feature, sequence], verbose=0)[0]
            top_indices = np.argsort(preds)[-beam_width:]

            for idx in top_indices:
                word = index_to_word(idx, tokenizer)
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

beam_width = st.slider(
    "Beam width",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    help="Use 1 for greedy decoding or higher values for beam search."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        image_feature = extract_features(image)

        if beam_width == 1:
            generated_caption = greedy_decode(
                caption_model,
                image_feature,
                tokenizer,
                max_length
            )
        else:
            generated_caption = beam_search_decode(
                caption_model,
                image_feature,
                tokenizer,
                max_length,
                beam_width=beam_width
            )

        final_caption = clean_caption(generated_caption)

    st.success("Caption generated!")
    st.subheader("Generated Caption:")
    st.write(final_caption)
