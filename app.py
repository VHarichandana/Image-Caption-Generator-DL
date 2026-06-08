import os
import streamlit as st
import pickle
import numpy as np
import matplotlib.cm as cm

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.models.attention_model import BahdanauAttention


st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and let AI generate a caption.")


MAX_LENGTH = 34
BEST_MODEL_PATH = "saved_models/best_attention_model.keras"
FALLBACK_MODEL_PATH = "saved_models/attention_model.keras"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"


def get_model_path():
    if os.path.exists(BEST_MODEL_PATH):
        return BEST_MODEL_PATH
    return FALLBACK_MODEL_PATH


MODEL_PATH = get_model_path()


@st.cache_resource
def load_caption_model():
    model = load_model(
        MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False
    )
    return model


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_extractor():
    base_model = ResNet50(weights="imagenet")

    return Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer("conv5_block3_out").output
    )


caption_model = load_caption_model()
tokenizer = load_tokenizer()
feature_extractor = load_feature_extractor()


def build_attention_model(caption_model):
    attention_layer = caption_model.get_layer("bahdanau_attention")
    attention_output = attention_layer.output

    if isinstance(attention_output, (list, tuple)):
        attention_weights = attention_output[1]
    else:
        raise ValueError("Attention layer output not found.")

    return Model(
        inputs=caption_model.inputs,
        outputs=[caption_model.output, attention_weights]
    )


attention_model = build_attention_model(caption_model)


def extract_features(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = feature_extractor.predict(image, verbose=0)
    feature = feature.reshape((1, 49, 2048)).astype(np.float32)

    return feature


def build_index_to_word(tokenizer):
    return {
        index: word
        for word, index in tokenizer.word_index.items()
    }


def model_predict(model, image_feature, sequence):
    image_feature = np.asarray(image_feature, dtype=np.float32)
    sequence = np.asarray(sequence, dtype=np.int32)

    preds = model.predict(
        [image_feature, sequence],
        verbose=0
    )

    return preds


def attention_predict(model, image_feature, sequence):
    image_feature = np.asarray(image_feature, dtype=np.float32)
    sequence = np.asarray(sequence, dtype=np.int32)

    preds, attention_weights = model.predict(
        [image_feature, sequence],
        verbose=0
    )

    return preds, attention_weights


def greedy_decode(model, image_feature, tokenizer, index_to_word, max_length):
    caption = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        ).astype(np.int32)

        yhat = model_predict(
            model,
            image_feature,
            sequence
        )

        yhat = int(np.argmax(yhat))

        word = index_to_word.get(yhat)

        if word is None:
            break

        caption += " " + word

        if word == "endseq":
            break

    return caption


def has_repeated_ngram(tokens, n=2):
    if len(tokens) < n * 2:
        return False

    ngrams = []

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)

    return len(ngrams) != len(set(ngrams))


def beam_search_decode(
    model,
    image_feature,
    tokenizer,
    index_to_word,
    max_length,
    beam_width=3,
    length_penalty=0.7,
    repetition_penalty=1.2
):
    beams = [[["startseq"], 0.0]]
    completed = []

    for _ in range(max_length):
        all_candidates = []

        for beam_tokens, beam_score in beams:
            if beam_tokens[-1] == "endseq":
                completed.append([beam_tokens, beam_score])
                continue

            caption_so_far = " ".join(beam_tokens)

            sequence = tokenizer.texts_to_sequences(
                [caption_so_far]
            )[0]

            sequence = pad_sequences(
                [sequence],
                maxlen=max_length
            ).astype(np.int32)

            preds = model_predict(
                model,
                image_feature,
                sequence
            )[0]

            top_indices = np.argsort(preds)[-(beam_width * 5):][::-1]

            for idx in top_indices:
                idx = int(idx)

                word = index_to_word.get(idx)

                if word is None:
                    continue

                if word == "startseq":
                    continue

                new_tokens = beam_tokens + [word]

                if len(beam_tokens) > 1 and word == beam_tokens[-1]:
                    continue

                if has_repeated_ngram(new_tokens, n=2):
                    continue

                probability = preds[idx]

                if word in beam_tokens:
                    probability = probability / repetition_penalty

                new_score = beam_score + np.log(probability + 1e-10)
                normalized_score = new_score / (len(new_tokens) ** length_penalty)

                all_candidates.append(
                    [new_tokens, new_score, normalized_score]
                )

        if not all_candidates:
            break

        all_candidates.sort(
            key=lambda x: x[2],
            reverse=True
        )

        beams = [
            [tokens, raw_score]
            for tokens, raw_score, _ in all_candidates[:beam_width]
        ]

    completed.extend(beams)

    completed.sort(
        key=lambda x: x[1] / (len(x[0]) ** length_penalty),
        reverse=True
    )

    return " ".join(completed[0][0])


def clean_caption(caption):
    caption = caption.replace("startseq", "")
    caption = caption.replace("endseq", "")
    return caption.strip()


def calculate_bleu_scores(reference_caption, generated_caption):
    reference = [reference_caption.lower().split()]
    candidate = generated_caption.lower().split()

    smoothing = SmoothingFunction().method1

    bleu1 = sentence_bleu(
        reference,
        candidate,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smoothing
    )

    bleu2 = sentence_bleu(
        reference,
        candidate,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing
    )

    bleu3 = sentence_bleu(
        reference,
        candidate,
        weights=(0.33, 0.33, 0.33, 0),
        smoothing_function=smoothing
    )

    bleu4 = sentence_bleu(
        reference,
        candidate,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )

    return bleu1, bleu2, bleu3, bleu4


def get_attention_maps(attention_model, image_feature, tokenizer, raw_caption, max_length):
    tokens = raw_caption.split()
    attention_data = []

    for i in range(1, len(tokens)):
        word = tokens[i]

        if word in ["startseq", "endseq"]:
            continue

        caption_so_far = " ".join(tokens[:i])

        sequence = tokenizer.texts_to_sequences(
            [caption_so_far]
        )[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        ).astype(np.int32)

        _, attention_weights = attention_predict(
            attention_model,
            image_feature,
            sequence
        )

        attention_map = attention_weights.reshape(7, 7)

        attention_data.append(
            {
                "word": word,
                "attention_map": attention_map
            }
        )

    return attention_data


def create_attention_overlay(image, attention_map, alpha=0.45):
    original_image = image.convert("RGB")

    attention_map = attention_map - np.min(attention_map)
    attention_map = attention_map / (np.max(attention_map) + 1e-8)

    heatmap = cm.jet(attention_map)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    heatmap_image = Image.fromarray(heatmap)
    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.BICUBIC
    )

    original_array = np.array(original_image).astype(np.float32)
    heatmap_array = np.array(heatmap_image).astype(np.float32)

    overlay = original_array * (1 - alpha) + heatmap_array * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


index_to_word = build_index_to_word(tokenizer)

st.sidebar.header("Settings")

beam_width = st.sidebar.slider(
    "Beam width",
    min_value=1,
    max_value=6,
    value=1,
    step=1,
    help="1 = greedy decoding, 2–6 = beam search"
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

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

        attention_data = get_attention_maps(
            attention_model,
            image_feature,
            tokenizer,
            raw_caption,
            MAX_LENGTH
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Caption Generator",
            "Attention Heatmap",
            "BLEU Score",
            "Limitations"
        ]
    )

    with tab1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.success("Caption generated!")

        st.subheader("Generated Caption")
        st.write(final_caption)

    with tab2:
        st.subheader("Attention Heatmap")

        st.write(
            "Select a generated word to see which image regions the attention layer focused on."
        )

        if attention_data:
            generated_words = [
                item["word"]
                for item in attention_data
            ]

            selected_word = st.selectbox(
                "Select word",
                generated_words
            )

            selected_item = next(
                item
                for item in attention_data
                if item["word"] == selected_word
            )

            overlay = create_attention_overlay(
                image,
                selected_item["attention_map"]
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    overlay,
                    caption=f"Attention Heatmap for: {selected_word}",
                    use_container_width=True
                )

            st.caption(
                "Brighter regions indicate stronger attention for the selected generated word."
            )

        else:
            st.info("Attention heatmap could not be generated for this caption.")

    with tab3:
        st.subheader("BLEU Score Evaluation")

        reference_caption = st.text_area(
            "Reference caption",
            placeholder="Example: a dog is running on the grass"
        )

        if reference_caption.strip():
            bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(
                reference_caption,
                final_caption
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("BLEU-1", f"{bleu1:.4f}")
            col2.metric("BLEU-2", f"{bleu2:.4f}")
            col3.metric("BLEU-3", f"{bleu3:.4f}")
            col4.metric("BLEU-4", f"{bleu4:.4f}")

            st.caption(
                "BLEU compares the generated caption with the reference caption. "
                "Higher score means stronger word overlap with the reference."
            )

        else:
            st.info(
                "Enter a reference caption to calculate BLEU score. "
                "Without a reference caption, BLEU cannot be calculated for custom images."
            )

    with tab4:
        st.subheader("Limitations")

        st.markdown(
            """
            - The model was trained on the small Flickr8k dataset.
            - It may generate incorrect captions for complex or unfamiliar images.
            - The vocabulary is limited to the training dataset.
            - Captions may sometimes become repetitive.
            - Beam search may not always perform better than greedy decoding for this trained model.
            - BLEU score needs a correct reference caption for comparison.
            """
        )

else:
    st.info("Upload an image to generate a caption.")
