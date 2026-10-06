import gradio as gr
from inference import predict_text_only, predict_competition

CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.hero {padding: 1.3rem 1.4rem; border: 1px solid var(--border-color-primary);
       border-radius: 18px; margin-bottom: 1rem;}
.metric-card {border: 1px solid var(--border-color-primary);
              border-radius: 14px; padding: 1rem; min-height: 108px;}
.small-note {opacity: .8; font-size: .92rem;}
@media (max-width: 768px) {
  .gradio-container {padding-left: 10px !important; padding-right: 10px !important;}
  .hero {padding: 1rem;}
}
"""


def _to_ui(result):
    return (
        f"Category {result['prediction']}",
        f"{result['confidence'] * 100:.2f}%",
        result["probabilities"],
    )


def run_text_model(comment):
    try:
        return _to_ui(predict_text_only(comment))
    except Exception as exc:
        raise gr.Error(str(exc))


def run_competition_model(
    comment, upvote, downvote, emoticon_1, emoticon_2, emoticon_3,
    race, religion, gender, disability
):
    try:
        result = predict_competition(
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
        return _to_ui(result)
    except Exception as exc:
        raise gr.Error(str(exc))


with gr.Blocks(
    title="Comment Category Prediction",
    #css=CUSTOM_CSS,
    #theme=gr.themes.Soft(),
) as demo:

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
            gr.Markdown("### 0.82365\n**Kaggle leaderboard score**  \nOriginal competition solution")
        with gr.Column(elem_classes="metric-card"):
            gr.Markdown("### 0.8024\n**Validation Macro F1**  \nText + metadata SGD")
        with gr.Column(elem_classes="metric-card"):
            gr.Markdown("### 0.7047\n**Validation Macro F1**  \nText-only deployment model")

    with gr.Tabs():

        with gr.Tab("🚀 Live Text Demo"):
            gr.Markdown("""
### Try the deployment-friendly model

Paste any comment below. This model uses **text only**, so no competition-specific
metadata is required.

> Categories 0–3 are anonymized labels. Probability reflects the model's confidence
> within those learned classes, not real-world semantic certainty.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    text_comment = gr.Textbox(
                        label="Comment",
                        placeholder="Type or paste a comment...",
                        lines=8,
                    )
                    text_predict_btn = gr.Button("Predict with Text Model", variant="primary")

                with gr.Column(scale=1):
                    text_category = gr.Textbox(label="Predicted Category", interactive=False)
                    text_confidence = gr.Textbox(label="Model Confidence", interactive=False)
                    text_probabilities = gr.Label(label="Class Probabilities", num_top_classes=4)

            gr.Examples(
                examples=[
                    ["hello"],
                    ["I completely disagree with this!"],
                    ["This is a much longer comment explaining my opinion..."],
                    ["😂😂😂"],
                ],
                inputs=[text_comment],
            )

            gr.Markdown("""
<div class="small-note">

**Why a separate model?** The competition dataset contains useful structured
metadata. This text-only version makes the public demo usable without inventing
hidden or unavailable metadata.

</div>
""")

        with gr.Tab("🧪 Competition Model"):
            gr.Markdown("""
### Original text + metadata pipeline

This version most closely represents the competition system and its stronger
validation performance. Anonymous hidden features and temporal variables use stable
training-distribution defaults; interpretable metadata can be adjusted below.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    comp_comment = gr.Textbox(
                        label="Comment",
                        placeholder="Enter a comment...",
                        lines=7,
                    )

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
                                ["unknown", "none", "white", "black", "other", "asian", "latino"],
                                value="unknown",
                                label="Race",
                            )
                            religion = gr.Textbox(value="unknown", label="Religion")
                            gender = gr.Textbox(value="unknown", label="Gender")

                        disability = gr.Checkbox(value=False, label="Disability indicator")

                    comp_predict_btn = gr.Button(
                        "Predict with Competition Model",
                        variant="primary",
                    )

                with gr.Column(scale=1):
                    comp_category = gr.Textbox(label="Predicted Category", interactive=False)
                    comp_confidence = gr.Textbox(label="Model Confidence", interactive=False)
                    comp_probabilities = gr.Label(label="Class Probabilities", num_top_classes=4)

            gr.Markdown("""
<div class="small-note">

**Deployment note:** `if_1`, `if_2`, month, hour and weekend status are hidden
because they are not meaningful public inputs. They are fixed to defaults derived
from the training distribution to avoid inference-distribution shift.

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
The target is imbalanced, so Macro F1 gives every class equal importance rather
than letting the majority class dominate the metric.

### Deployment findings
- The competition model was serialized with `cloudpickle` and verified with
  **identical predictions before and after loading**.
- Arbitrary defaults for hidden features caused severe inference-distribution shift.
- The final competition inference path uses defaults derived from training data.
- The text-only model scores lower, showing that structured metadata carried
  meaningful predictive signal in the original task.

### Architecture

```text
Live Text Demo
Comment
  ↓
Word + character TF-IDF
  ↓
SGDClassifier (log_loss)
  ↓
Category probabilities

Competition Model
Comment + interpretable metadata
  ↓
Feature engineering
  ↓
Fitted preprocessing pipeline
  ↓
SGDClassifier (log_loss)
  ↓
Category probabilities
```
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

### Interpretation note
The target classes remain **Category 0–3** because the competition labels are
anonymized. The application does not invent semantic meanings for them.
""")

    text_predict_btn.click(
        fn=run_text_model,
        inputs=[text_comment],
        outputs=[text_category, text_confidence, text_probabilities],
    )

    comp_predict_btn.click(
        fn=run_competition_model,
        inputs=[
            comp_comment, upvote, downvote, emoticon_1, emoticon_2, emoticon_3,
            race, religion, gender, disability
        ],
        outputs=[comp_category, comp_confidence, comp_probabilities],
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )



