---
title: Comment Category Prediction
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Comment Category Prediction

A public ML demo built from a multiclass Kaggle-style comment classification project.

## Model

The deployed model is an SGDClassifier (`loss="log_loss"`) wrapped in the fitted
scikit-learn preprocessing pipeline and serialized with `cloudpickle`.

- Validation Macro F1: **0.8024**
- Kaggle leaderboard score: **0.82365**
- Classes: **0, 1, 2, 3** (anonymized competition categories)
- Saved model size: approximately **2.84 MB**

## Before deploying

Copy your verified model file into:

```text
models/sgd_classifier_pipeline.pkl
```

The model was verified before deployment with identical predictions before and after
serialization.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL printed in your terminal.

## Project structure

```text
.
├── app.py
├── inference.py
├── feature_engineering.py
├── models/
│   └── sgd_classifier_pipeline.pkl
├── requirements.txt
└── README.md
```

## Deployment behavior

The competition model was trained using both text and metadata. The public demo exposes
meaningful optional metadata such as votes and emoticons. Anonymous competition features
(`if_1`, `if_2`) and temporal fields use stable defaults derived from the training
distribution so the app does not fabricate arbitrary live values.

The categories are intentionally displayed as Category 0–3 because the competition
labels are anonymized.
