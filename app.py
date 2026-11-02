"""
Food Recommendation Classifier
Run with:
    pip install streamlit pandas numpy scikit-learn==1.6.1 joblib
    streamlit run app.py

Keep food_recommendation_model.joblib in the same folder as this file.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "food_recommendation_model.joblib"

st.set_page_config(
    page_title="Food Recommendation",
    page_icon="🍽️",
    layout="wide",
)

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place food_recommendation_model.joblib beside app.py."
        )
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "scaler", "le_target", "feature_cols"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model file is missing required items: {', '.join(sorted(missing))}")
    return artifact

try:
    artifact = load_artifact()
except Exception as exc:
    st.error(f"Could not load the recommendation model: {exc}")
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
label_encoder = artifact["le_target"]
feature_cols = artifact["feature_cols"]

# The saved model was trained with 17 numeric columns followed by one-hot
# encoded categorical columns. Keep this order exactly as it was during training.
numeric_cols = feature_cols[: scaler.n_features_in_]
categorical_cols = feature_cols[scaler.n_features_in_ :]

def category_options(prefix):
    return [
        col[len(prefix):]
        for col in categorical_cols
        if col.startswith(prefix)
    ]

def readable(text):
    return " ".join(str(text).replace("\n", " ").split()).strip(" .")

st.title("🍽️ Food Recommendation")
st.write(
    "Enter the food's nutrition information and select its product and label details. "
    "The trained model will classify it as **High** or **Low**."
)

with st.expander("About this model", expanded=False):
    st.write(
        "This app uses the supplied Random Forest classifier, its saved StandardScaler, "
        "and its saved target LabelEncoder. The input columns and their order are taken "
        "directly from the model file."
    )

st.subheader("Nutrition information")
st.caption("Enter values using the same units and definitions as the data used to train the model.")

numeric_labels = [
    "Average portion weight",
    "Number of servings",
    "Carbohydrate",
    "Protein",
    "Total fat",
    "Fat % of RDA",
    "Trans fat",
    "Trans fat % of RDA",
    "MUFA",
    "PUFA",
    "Saturated fat",
    "Saturated fat % of RDA",
    "Sodium",
    "Sodium (second recorded field)",
    "Sodium % of RDA",
    "Sugar",
    "Sugar % of RDA",
]

numeric_values = {}
columns = st.columns(3)
for i, (col_name, label) in enumerate(zip(numeric_cols, numeric_labels)):
    default = float(scaler.mean_[i]) if hasattr(scaler, "mean_") else 0.0
    with columns[i % 3]:
        numeric_values[col_name] = st.number_input(
            label,
            value=default,
            step=0.1,
            format="%.3f",
            key=f"numeric_{i}",
            help=f"Model input field: {readable(col_name)}",
        )

st.subheader("Product and label details")
st.caption("Choose one option in each category. Leave a category as “Not specified” if it does not apply.")

category_groups = [
    ("Products_", "Product"),
    ("Allergens_", "Allergen information"),
    ("MSG Info_", "MSG information"),
    ("Caffeine and Artificial Sweetene", "Caffeine / sweetener information"),
]

selected_categories = {}
for prefix, label in category_groups:
    options = category_options(prefix)
    if not options:
        continue
    # A neutral option is available when the training data included an Unknown category.
    unknown = next((x for x in options if x.lower() == "unknown"), None)
    display_options = (["Not specified"] if unknown is None else [unknown])
    display_options += [x for x in options if x != unknown]

    selected = st.selectbox(
        label,
        options=display_options,
        format_func=lambda value: "Not specified" if value == "Not specified" else readable(value),
        key=f"category_{prefix}",
    )
    selected_categories[prefix] = selected

if st.button("Get recommendation", type="primary", use_container_width=True):
    # Build the exact feature vector expected by the saved classifier.
    row = {col: 0.0 for col in feature_cols}

    numeric_frame = pd.DataFrame(
        [[numeric_values[col] for col in numeric_cols]],
        columns=numeric_cols,
    )
    scaled_numeric = scaler.transform(numeric_frame)

    for i, col in enumerate(numeric_cols):
        row[col] = float(scaled_numeric[0, i])

    for prefix, selected in selected_categories.items():
        if selected == "Not specified":
            continue
        encoded_col = prefix + selected
        if encoded_col in row:
            row[encoded_col] = 1.0

    model_input = pd.DataFrame([[row[col] for col in feature_cols]], columns=feature_cols)

    try:
        prediction = model.predict(model_input)[0]
        predicted_label = label_encoder.inverse_transform([int(prediction)])[0]

        st.divider()
        st.subheader("Prediction")
        if str(predicted_label).lower() == "high":
            st.success(f"Model classification: {predicted_label}")
        else:
            st.info(f"Model classification: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            probability_labels = label_encoder.inverse_transform(model.classes_)
            probability_df = pd.DataFrame({
                "Classification": probability_labels,
                "Probability": probabilities,
            }).sort_values("Probability", ascending=False)
            st.write("Prediction confidence")
            st.dataframe(
                probability_df.assign(
                    Probability=probability_df["Probability"].map(lambda x: f"{x:.1%}")
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption("Probabilities are model estimates, not guarantees.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Important: This is a machine-learning classification, not medical or dietary advice. "
    "Check ingredient and allergen information directly before making food choices."
)
