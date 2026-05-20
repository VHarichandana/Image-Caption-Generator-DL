# run this as a one-time script: resave_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.models.attention_model import BahdanauAttention

# load old model using custom_objects
model = load_model(
    "saved_models/caption_model.keras",
    custom_objects={'BahdanauAttention': BahdanauAttention}
)

# resave — now the decorator is registered so Keras
# writes the correct class path into the file
model.save("saved_models/caption_model.keras")

print("Model resaved successfully.")