import os

from src.config import IMAGES_DIR

images_path = IMAGES_DIR

images = os.listdir(images_path)

print("Total Images:", len(images))
print(images[:5])