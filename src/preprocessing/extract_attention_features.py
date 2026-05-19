import os
import pickle
import numpy as np

from tqdm import tqdm

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.models import Model

from src.config import IMAGES_DIR, SAVED_MODELS_DIR, ATTENTION_FEATURES_FILE


# -------------------------------------------------
# IMAGE DIRECTORY
# -------------------------------------------------

images_path = IMAGES_DIR


# -------------------------------------------------
# LOAD RESNET50
# -------------------------------------------------

base_model = ResNet50(
    weights='imagenet'
)

# IMPORTANT:
# NO GLOBAL AVERAGE POOLING

feature_extractor = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-3].output
)

print(feature_extractor.output_shape)


# -------------------------------------------------
# EXTRACT FEATURES
# -------------------------------------------------

features = {}

images = os.listdir(images_path)

print("\nExtracting Attention Features...\n")

for image_name in tqdm(images):

    image_path = os.path.join(
        images_path,
        image_name
    )

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(image)

    feature = feature_extractor.predict(
        image,
        verbose=0
    )

    # shape becomes:
    # (1, 7, 7, 2048)

    feature = np.reshape(
        feature,
        (49, 2048)
    )

    features[image_name] = feature


# -------------------------------------------------
# SAVE FEATURES
# -------------------------------------------------

os.makedirs(
    SAVED_MODELS_DIR,
    exist_ok=True
)

with open(
    ATTENTION_FEATURES_FILE,
    "wb"
) as f:

    pickle.dump(
        features,
        f
    )

print("\nAttention features saved successfully!")