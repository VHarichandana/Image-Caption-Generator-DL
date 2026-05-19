from src.models.caption_model import build_caption_model

VOCAB_SIZE = 8768
MAX_LENGTH = 34

model = build_caption_model(
    VOCAB_SIZE,
    MAX_LENGTH
)

model.summary()