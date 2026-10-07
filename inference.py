from pathlib import Path
import cloudpickle
import numpy as np
from feature_engineering import build_competition_features

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

COMPETITION_MODEL_PATH = MODELS_DIR / "sgd_classifier_pipeline.pkl"
TEXT_MODEL_PATH = MODELS_DIR / "text_only_sgd_pipeline.pkl"
EXPECTED_CLASSES = np.array([0, 1, 2, 3])

def _load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path.name}")
    with path.open("rb") as f:
        model = cloudpickle.load(f)
    if not np.array_equal(model.classes_, EXPECTED_CLASSES):
        raise RuntimeError(f"Unexpected classes in {path.name}: {model.classes_}")
    return model

competition_model = _load_model(COMPETITION_MODEL_PATH)
text_model = _load_model(TEXT_MODEL_PATH)

def _format_result(classes, probabilities):
    best_idx = int(np.argmax(probabilities))
    return {
        "prediction": int(classes[best_idx]),
        "confidence": float(probabilities[best_idx]),
        "probabilities": {
            f"Category {int(c)}": float(p)
            for c, p in zip(classes, probabilities)
        }
    }

def predict_text_only(comment):
    if comment is None or not str(comment).strip():
        raise ValueError("Please enter a comment before predicting.")
    proba = text_model.predict_proba([str(comment)])[0]
    return _format_result(text_model.classes_, proba)

def predict_competition(
    comment, upvote=1, downvote=0,
    emoticon_1=0, emoticon_2=0, emoticon_3=0,
    race="unknown", religion="unknown", gender="unknown",
    disability=False
):
    if comment is None or not str(comment).strip():
        raise ValueError("Please enter a comment before predicting.")
    features = build_competition_features(
        comment, upvote, downvote,
        emoticon_1, emoticon_2, emoticon_3,
        race, religion, gender, disability
    )
    proba = competition_model.predict_proba(features)[0]
    return _format_result(competition_model.classes_, proba)

