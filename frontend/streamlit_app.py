"""
Streamlit frontend for the AG News classifier, calling the FastAPI backend
and visualizing class probabilities for each input text.
"""

import requests
import streamlit as st
import pandas as pd
import altair as alt

API_BASE_URL = "http://127.0.0.1:8000"


def call_api(texts, api_base_url: str, endpoint: str):
    """Call a FastAPI prediction endpoint with a list of texts."""
    url = api_base_url.rstrip("/") + endpoint
    payload = {"texts": texts}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def main():
    """Render the Streamlit UI and handle user interaction for classification."""
    st.title("AG News Topic Classifier")
    st.caption("NOTE: Use the sidebar to swap models")

    st.markdown(
        "Enter one or more news headlines or short articles below, "
        "with **one entry per line**. The app will classify each into "
        "one of the four AG News categories."
    )

    # Sidebar: API base URL + model selection
    api_base_url = st.sidebar.text_input("API base URL", value=API_BASE_URL)

    model_choice = st.sidebar.selectbox(
        "Model",
        [
            "Baseline (TF-IDF + Logistic Regression)",
            "Transformer (DistilBERT)",
        ],
    )

    if model_choice.startswith("Baseline"):
        endpoint = "/predict"
    else:
        endpoint = "/predict-transformer"

    st.subheader("Input")
    raw_text = st.text_area(
        "News texts (one per line)",
        height=200,
        placeholder=(
            "Example:\n"
            "Stock markets rally as tech companies report strong earnings.\n"
            "Local team wins championship finals after dramatic overtime."
        ),
    )

    if st.button("Classify"):
        texts = [line.strip() for line in raw_text.split("\n") if line.strip()]

        if not texts:
            st.warning("Please enter at least one non-empty line of text.")
            return

        try:
            with st.spinner(f"Calling the API ({model_choice})..."):
                data = call_api(texts, api_base_url, endpoint)

            if not data.get("predictions"):
                st.error("No predictions returned from the API.")
                return

            predictions = data["predictions"]
            labels = ["World", "Sports", "Business", "Sci/Tech"]

            # Fixed color mapping to keep label colors consistent across charts
            color_map = {
                "World": "#2ca02c",
                "Sports": "#ff7f0e",
                "Business": "#d62728",
                "Sci/Tech": "#1f77b4",
            }

            st.subheader("Predictions")

            for i, pred in enumerate(predictions, start=1):
                st.markdown(f"### Sentence {i}")
                st.write(f"**Input text:** {pred['text'].strip()}")
                st.write(
                    f"**Predicted label ({model_choice}):** "
                    f"{pred['label']} (id: {pred['label_id']})"
                )

                probs_df = pd.DataFrame(
                    {
                        "label": labels,
                        "probability": pred["probs"],
                    }
                )

                chart = (
                    alt.Chart(probs_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", title="Category"),
                        y=alt.Y("probability:Q", title="Probability"),
                        color=alt.Color(
                            "label:N",
                            scale=alt.Scale(
                                domain=labels, range=list(color_map.values())
                            ),
                            legend=None,
                        ),
                        tooltip=["label", "probability"],
                    )
                    .properties(height=250)
                )

                st.altair_chart(chart, width="stretch")
                st.markdown("---")

        except requests.RequestException as e:
            st.error(f"Error calling API: {e}")


if __name__ == "__main__":
    main()
