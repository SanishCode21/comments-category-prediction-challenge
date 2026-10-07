import gradio as gr
from inference import predict_text_only, predict_competition

DEVELOPER_NAME = "Sanish Kumar"

# Replace with your exact public URLs before publishing.
LINKEDIN_URL = "https://www.linkedin.com/in/sanish-kumar-singh-163679289"
GITHUB_URL = "https://github.com/SanishCode21/comments-category-prediction-challenge.git"
COLAB_LINK = "https://colab.research.google.com/drive/1rnMK-AG9wUr937dHUoPtxNxZOYb4ygif?usp=sharing"
KAGGLE_URL = "https://www.kaggle.com/code/sanishkumarsingh"
KAGGLE_COMPETITION = "https://www.kaggle.com/competitions/comment-category-prediction-challenge/overview"
LIVE_APP_URL = "https://sanishkumarsingh-comment-category-prediction-challenge.hf.space/"

CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.hero {padding: 1.25rem 1.4rem; border: 1px solid var(--border-color-primary);
       border-radius: 18px; margin-bottom: 1rem;}
.metric-card {border: 1px solid var(--border-color-primary);
              border-radius: 14px; padding: 1rem; min-height: 108px;}
.small-note {opacity: .8; font-size: .92rem;}
@media (max-width: 768px) {
  .gradio-container {padding-left: 10px !important; padding-right: 10px !important;}
  .hero {padding: 1rem;}
}
"""

VALIDATION_EXAMPLES = [
    "the us is an international bully and needs its physical threat. of course, the us will not seriously consider disarming. we are more concerned in making sure that other nations do not acqiure our abilities to kill.",
    "can trupm stop putting his foot in his mouth and stay out of it?",
    "good luck with my-way-or-the-highway stevenson there's a reason she's no longer in jeffco.",
    "because he thought dusg they were attending services there. he posted threats to his mother in law that very morning. he was after them, and their friends. a nut-job that should have never been able to own a firearm. but the nra, congress, texas and the air force made sure it was an option available to him. reaping what you have sown.",
    'why is it, that the native corporation have become million dollar entities? those "jokers" must be doing something right to offer dividends year after year.. put that in your hat and eat it robert.'
]

def _to_ui(result):
    return (
        f"Category {result['prediction']}",
        f"{result['confidence'] * 100:.2f}%",
        result["probabilities"]
    )


def run_text_model(comment):
    if comment is None or not str(comment).strip():
        return (
            "No prediction",
            "—",
            {
                "Category 0": 0.0,
                "Category 1": 0.0,
                "Category 2": 0.0,
                "Category 3": 0.0,
            },
        )

    try:
        return _to_ui(predict_text_only(comment))
    except Exception as exc:
        return (
            "Prediction error",
            str(exc),
            {
                "Category 0": 0.0,
                "Category 1": 0.0,
                "Category 2": 0.0,
                "Category 3": 0.0,
            },
        )


def run_competition_model(
    comment,
    upvote,
    downvote,
    emoticon_1,
    emoticon_2,
    emoticon_3,
    race,
    religion,
    gender,
    disability,
):
    if comment is None or not str(comment).strip():
        return (
            "No prediction",
            "Please enter a comment.",
            {
                "Category 0": 0.0,
                "Category 1": 0.0,
                "Category 2": 0.0,
                "Category 3": 0.0,
            },
        )

    try:
        result = predict_competition(
            comment,
            upvote,
            downvote,
            emoticon_1,
            emoticon_2,
            emoticon_3,
            race,
            religion,
            gender,
            disability,
        )

        return _to_ui(result)

    except Exception as exc:
        return (
            "Prediction error",
            str(exc),
            {
                "Category 0": 0.0,
                "Category 1": 0.0,
                "Category 2": 0.0,
                "Category 3": 0.0,
            },
        )

with gr.Blocks(title="Comment Category Prediction") as demo:
    gr.Markdown("""
<div class="hero">

# 💬 Comment Category Prediction

**End-to-end multiclass ML system - competition modeling + deployment adaptation**

This project classifies comments into four **anonymized competition categories**.
The original solution combines text with structured metadata; a second text-only
model was trained specifically for public interaction.

</div>
""")

    with gr.Row():
        with gr.Column(elem_classes="metric-card"):
            gr.Markdown("### 0.82365\n**Kaggle leaderboard score**")
        with gr.Column(elem_classes="metric-card"):
            gr.Markdown("### 0.8024\n**Validation Macro F1**  \nText + metadata SGD")
        with gr.Column(elem_classes="metric-card"):
            gr.Markdown("### 0.7047\n**Validation Macro F1**  \nText-only model")

    with gr.Tabs():

        with gr.Tab("🚀 Live Text Demo"):
            gr.Markdown("""
### Try the deployment-friendly model
Paste any comment below. This model uses **text only**, so no competition-specific
metadata is required.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    text_comment = gr.Textbox(
                        label="Comment",
                        placeholder="Enter a comment before clicking Predict...",
                        lines=8
                    )
                    text_predict_btn = gr.Button("Predict with Text Model", variant="primary")
                with gr.Column(scale=1):
                    text_category = gr.Textbox(label="Predicted Category", interactive=False)
                    text_confidence = gr.Textbox(label="Model Confidence", interactive=False)
                    text_probabilities = gr.Label(label="Class Probabilities", num_top_classes=4)

            gr.Examples(
                examples=[[x] for x in VALIDATION_EXAMPLES],
                inputs=[text_comment],
                label="Real validation comments"
            )

            gr.Markdown("""
<div class="small-note">
These examples are taken from the validation set so the demo stays representative
of the original data distribution.
</div>
""")

        with gr.Tab("Competition Model"):
            gr.Markdown("""
### Original text + metadata pipeline
This version most closely represents the competition system and its stronger
validation performance.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    comp_comment = gr.Textbox(label="Comment", lines=7)

                    with gr.Accordion("Optional metadata", open=False):
                        with gr.Row():
                            upvote = gr.Number(label="Upvotes", value=1, minimum=0, precision=0)
                            downvote = gr.Number(label="Downvotes", value=0, minimum=0, precision=0)

                        with gr.Row():
                            emoticon_1 = gr.Number(label="Emoticon 1", value=0, minimum=0, precision=0)
                            emoticon_2 = gr.Number(label="Emoticon 2", value=0, minimum=0, precision=0)
                            emoticon_3 = gr.Number(label="Emoticon 3", value=0, minimum=0, precision=0)

                        with gr.Row():
                            race = gr.Dropdown(
                                ["unknown","none","white","black","other","asian","latino"],
                                value="unknown", label="Race"
                            )
                            religion = gr.Textbox(value="unknown", label="Religion")
                            gender = gr.Textbox(value="unknown", label="Gender")

                        disability = gr.Checkbox(value=False, label="Disability indicator")

                    comp_predict_btn = gr.Button("Predict with Competition Model", variant="primary")

                with gr.Column(scale=1):
                    comp_category = gr.Textbox(label="Predicted Category", interactive=False)
                    comp_confidence = gr.Textbox(label="Model Confidence", interactive=False)
                    comp_probabilities = gr.Label(label="Class Probabilities", num_top_classes=4)

            gr.Markdown("""
<div class="small-note">
Hidden competition variables and temporal fields use stable defaults derived from
the training distribution to avoid inference-distribution shift.
</div>
""")

        with gr.Tab("📊 Model Insights"):
            gr.Markdown("""
## Model comparison

| Model | Inputs | Validation Macro F1 | Purpose |
|---|---|---:|---|
| **SGDClassifier** | Text + metadata | **0.8024** | Main competition model |
| **Text-only SGDClassifier** | Comment only | **0.7047** | Public interaction model |
| Kaggle submission | Competition pipeline | **0.82365** | Leaderboard result |

### Why Macro F1?
The target is imbalanced, so Macro F1 gives every class equal importance.

### Production observations
- Serialized models were verified with identical predictions after reload.
- Arbitrary hidden-feature defaults caused severe deployment distribution shift.
- Stable training-derived defaults fixed that issue for the competition model.
- The text-only model is easier to use publicly but scores lower, showing that
  structured metadata carried meaningful predictive signal.
""")

        with gr.Tab("Development Journey"):
            gr.Markdown("""
## How this project evolved

- Performed EDA on text, votes, emoticons, demographic fields, date/time features,
  hidden variables, and post-level behavior.
- Engineered text statistics, engagement, vote, interaction, temporal, and
  categorical features.
- Compared Logistic Regression, SGDClassifier, RidgeClassifier, Random Forest,
  and earlier LightGBM experiments.
- Used Macro F1 because the target classes are imbalanced.
- Tested weighted probability ensembling.
- Diagnosed a major submission issue caused by inconsistent train/test dataframe
  column ordering.
- Improved the leaderboard score from about **0.3124 to 0.82365** after fixing
  inference alignment.
- Verified serialized pipelines with identical predictions before/after loading.
- Added a second text-only model for public inference after detecting that hidden
  metadata defaults could distort live predictions.

### Why two models?
**Competition model:** best representation of the original project.  
**Text-only model:** deployment adaptation for simple public interaction.
""")

        with gr.Tab("👨‍💻 Developer"):
            gr.Markdown(f"""
## Developed by {DEVELOPER_NAME}

I built this project as part of my machine-learning portfolio to demonstrate the
full lifecycle of a practical classification system: exploration, feature
engineering, model comparison, evaluation, debugging, serialization, and deployment.

### Profiles & project links
- **LinkedIn:** [{DEVELOPER_NAME}]({LINKEDIN_URL})
- **GitHub:** [Project repository]({GITHUB_URL})
- **Kaggle:** [Competition / notebook]({KAGGLE_URL})
- **Live app:** [Hugging Face Space]({LIVE_APP_URL})

> Replace the placeholder URLs at the top of `app.py` before publishing.
""")

        with gr.Tab("🧭 Project Overview"):
            gr.Markdown("""
## What this project demonstrates

**Problem:** multiclass comment-category prediction with anonymized labels and
class imbalance.

**Workflow:** EDA → feature engineering → preprocessing → model comparison →
validation → leaderboard submission → serialization → deployment.

**Engineering lessons:** consistent train/test schemas, robust inference defaults,
distribution-shift debugging, serialization verification, and adapting a
competition model for public use.

The target labels remain **Category 0–3** because the competition categories are
anonymized.
""")

    text_predict_btn.click(
        fn=run_text_model,
        inputs=[text_comment],
        outputs=[text_category, text_confidence, text_probabilities]
    )

    comp_predict_btn.click(
        fn=run_competition_model,
        inputs=[
            comp_comment, upvote, downvote, emoticon_1, emoticon_2, emoticon_3,
            race, religion, gender, disability
        ],
        outputs=[comp_category, comp_confidence, comp_probabilities]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)


