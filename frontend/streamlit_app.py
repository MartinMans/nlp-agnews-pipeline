import requests
import streamlit as st
import pandas as pd
import altair as alt

# Default URL of the FastAPI service
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"


def call_api(texts, api_url: str):
    """Call the FastAPI /predict endpoint with a list of texts."""
    payload = {"texts": texts}
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    return response.json()


def main():
    st.title("AG News Topic Classifier")

    st.markdown(
        "Enter one or more news headlines or short articles below, "
        "with **one entry per line**. The app will classify each into "
        "one of the four AG News categories."
    )

    # Sidebar: API URL (can use to change URL later)
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

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
        # Split input lines into list of texts
        texts = [line.strip() for line in raw_text.split("\n") if line.strip()]

        if not texts:
            st.warning("Please enter at least one non-empty line of text.")
            return

        try:
            with st.spinner("Calling the API..."):
                data = call_api(texts, api_url)

            if not data.get("predictions"):
                st.error("No predictions returned from the API.")
                return

            predictions = data["predictions"]
            labels = ["World", "Sports", "Business", "Sci/Tech"]

            # Define unique colors for each label
            color_map = {
                "World": "#2ca02c",     # Green
                "Sports": "#ff7f0e",    # Orange
                "Business": "#d62728",  # Red
                "Sci/Tech": "#1f77b4",  # Blue
            }

            st.subheader("Predictions")

            for i, pred in enumerate(predictions, start=1):
                snippet = pred['text'][:40] + ("..." if len(pred['text']) > 40 else "")
                st.markdown(f"#### {snippet}")
                st.write(f"**Predicted label:** {pred['label']} (id: {pred['label_id']})")

                # Build DataFrame for probabilities
                probs_df = pd.DataFrame({
                    "label": labels,
                    "probability": pred["probs"]
                })

                # Create a bar chart with unique colors
                chart = (
                    alt.Chart(probs_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", title="Category"),
                        y=alt.Y("probability:Q", title="Probability"),
                        color=alt.Color("label:N",
                                        scale=alt.Scale(domain=labels, range=list(color_map.values())),
                                        legend=None),
                        tooltip=["label", "probability"]
                    )
                    .properties(height=250)
                )

                st.altair_chart(chart, width="stretch")
                st.markdown("---")

        except requests.RequestException as e:
            st.error(f"Error calling API: {e}")


if __name__ == "__main__":
    main()
