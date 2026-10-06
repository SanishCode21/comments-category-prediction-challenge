from pathlib import Path
import cloudpickle
import numpy as np

from feature_engineering import build_competition_features

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

COMPETITION_MODEL_PATH = MODELS_DIR / "sgd_classifier_pipeline.pkl"
TEXT_MODEL_PATH = MODELS_DIR / "text_only_sgd_pipeline.pkl"

EXPECTED_CLASSES = np.array([0, 1, 2, 3])


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {path.name}. "
            "Place it inside the models/ directory."
        )

    with path.open("rb") as f:
        model = cloudpickle.load(f)

    if not np.array_equal(model.classes_, EXPECTED_CLASSES):
        raise RuntimeError(
            f"{path.name} has unexpected classes {model.classes_}; "
            f"expected {EXPECTED_CLASSES}."
        )

    return model


competition_model = _load_model(COMPETITION_MODEL_PATH)
text_model = _load_model(TEXT_MODEL_PATH)


def _format_result(classes, probabilities):
    best_idx = int(np.argmax(probabilities))
    prediction = int(classes[best_idx])
    confidence = float(probabilities[best_idx])

    probability_dict = {
        f"Category {int(label)}": float(prob)
        for label, prob in zip(classes, probabilities)
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
    }


def predict_text_only(comment):
    if comment is None or not str(comment).strip():
        raise ValueError("Please enter a comment before predicting.")

    probabilities = text_model.predict_proba([str(comment)])[0]
    return _format_result(text_model.classes_, probabilities)


def predict_competition(
    comment,
    upvote=1,
    downvote=0,
    emoticon_1=0,
    emoticon_2=0,
    emoticon_3=0,
    race="unknown",
    religion="unknown",
    gender="unknown",
    disability=False,
):
    if comment is None or not str(comment).strip():
        raise ValueError("Please enter a comment before predicting.")

    features = build_competition_features(
        comment=comment,
        upvote=upvote,
        downvote=downvote,
        emoticon_1=emoticon_1,
        emoticon_2=emoticon_2,
        emoticon_3=emoticon_3,
        race=race,
        religion=religion,
        gender=gender,
        disability=disability,
    )

    probabilities = competition_model.predict_proba(features)[0]
    return _format_result(competition_model.classes_, probabilities)

