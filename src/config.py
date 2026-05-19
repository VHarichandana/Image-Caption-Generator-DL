import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
ATTENTION_FEATURES_FILE = os.path.join(SAVED_MODELS_DIR, "attention_features.pkl")
IMAGE_FEATURES_FILE = os.path.join(SAVED_MODELS_DIR, "image_features.pkl")
TOKENIZER_FILE = os.path.join(SAVED_MODELS_DIR, "tokenizer.pkl")
BEST_MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "best_attention_model.keras")
ATTENTION_MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "attention_model.keras")
CAPTION_MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "caption_model.keras")
