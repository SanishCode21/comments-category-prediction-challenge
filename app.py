import gradio as gr

from inference import predict_comment


def run_prediction(
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
    try:
        result = predict_comment(
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

        predicted = f"Category {result['prediction']}"
        confidence = f"{result['confidence'] * 100:.2f}%"

        return predicted, confidence, result["probabilities"]

    except Exception as exc:
        raise gr.Error(str(exc))


with gr.Blocks(title="Comment Category Prediction") as demo:
    gr.Markdown(
        """
# Comment Category Prediction
### End-to-end multiclass ML demo

This project classifies comments into one of four **anonymized competition categories**
using TF-IDF text features, engineered metadata, and an SGD classifier.

**Validation Macro F1:** `0.8024` &nbsp; | &nbsp;
**Kaggle Leaderboard Score:** `0.82365`

The original competition used text together with metadata. For a convenient public
demo, anonymous competition features and time-related variables use stable defaults
derived from the training distribution.
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            comment = gr.Textbox(
                label="Comment",
                placeholder="Enter a comment to classify...",
                lines=7,
            )

            with gr.Accordion("Optional context", open=False):
                with gr.Row():
                    upvote = gr.Number(
                        label="Upvotes",
                        value=1,
                        minimum=0,
                        precision=0,
                    )
                    downvote = gr.Number(
                        label="Downvotes",
                        value=0,
                        minimum=0,
                        precision=0,
                    )

                with gr.Row():
                    emoticon_1 = gr.Number(
                        label="Emoticon 1",
                        value=0,
                        minimum=0,
                        precision=0,
                    )
                    emoticon_2 = gr.Number(
                        label="Emoticon 2",
                        value=0,
                        minimum=0,
                        precision=0,
                    )
                    emoticon_3 = gr.Number(
                        label="Emoticon 3",
                        value=0,
                        minimum=0,
                        precision=0,
                    )

                gr.Markdown(
                    "Demographic metadata is optional and defaults to `unknown`."
                )

                with gr.Row():
                    race = gr.Dropdown(
                        ["unknown", "none", "white", "black", "other", "asian", "latino"],
                        value="unknown",
                        label="Race",
                    )
                    religion = gr.Textbox(
                        value="unknown",
                        label="Religion",
                    )
                    gender = gr.Textbox(
                        value="unknown",
                        label="Gender",
                    )

                disability = gr.Checkbox(
                    value=False,
                    label="Disability indicator",
                )

            predict_btn = gr.Button(
                "Predict Category",
                variant="primary",
            )

        with gr.Column(scale=1):
            predicted_category = gr.Textbox(
                label="Predicted Category",
                interactive=False,
            )

            confidence = gr.Textbox(
                label="Model Confidence",
                interactive=False,
            )

            probabilities = gr.Label(
                label="Class Probabilities",
                num_top_classes=4,
            )

    gr.Examples(
        examples=[
            ["I completely disagree with this!", 1, 0, 0, 0, 0, "unknown", "unknown", "unknown", False],
            ["Thanks for explaining your point so clearly.", 3, 0, 0, 0, 0, "unknown", "unknown", "unknown", False],
            ["😂😂😂", 1, 0, 3, 0, 0, "unknown", "unknown", "unknown", False],
        ],
        inputs=[
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
        ],
    )

    gr.Markdown(
        """
---
### Model notes

- **Model:** SGDClassifier with `loss="log_loss"`
- **Features:** word/character TF-IDF + numerical/categorical engineered features
- **Metric:** Macro F1, chosen because the four classes are imbalanced
- **Target labels:** Categories 0–3 are anonymized competition labels
- **Deployment note:** the public quick demo uses training-distribution defaults for
  hidden competition features that have no meaningful user-facing interpretation.

This application is a portfolio demonstration of the complete ML workflow:
EDA → feature engineering → preprocessing → model comparison → validation →
serialization → live inference.
"""
    )

    predict_btn.click(
        fn=run_prediction,
        inputs=[
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
        ],
        outputs=[
            predicted_category,
            confidence,
            probabilities,
        ],
    )

if __name__ == "__main__":
    demo.launch()
