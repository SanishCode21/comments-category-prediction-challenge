from pathlib import Path
import cloudpickle
import numpy as np

from feature_engineering import build_features

MODEL_PATH = Path(__file__).parent / "models" / "sgd_classifier_pipeline.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Place sgd_classifier_pipeline.pkl inside the models/ folder."
    )

with MODEL_PATH.open("rb") as f:
    model = cloudpickle.load(f)

EXPECTED_CLASSES = np.array([0, 1, 2, 3])

if not np.array_equal(model.classes_, EXPECTED_CLASSES):
    raise RuntimeError(
        f"Unexpected model classes: {model.classes_}. "
        f"Expected {EXPECTED_CLASSES}."
    )


def predict_comment(
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
    """Return the predicted anonymized category and model probabilities."""
    if comment is None or not str(comment).strip():
        raise ValueError("Please enter a comment before predicting.")

    features = build_features(
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

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

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
