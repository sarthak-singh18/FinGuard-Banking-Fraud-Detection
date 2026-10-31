
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FinGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

MODEL_PATH = OUTPUTS / "final_fraud_model.pkl"
SUMMARY_PATH = OUTPUTS / "experiment_summary.json"
THRESHOLD_PATH = OUTPUTS / "decision_threshold.json"


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(SUMMARY_PATH, "r") as file:
        summary = json.load(file)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = json.load(file)["threshold"]

    return model, summary, threshold


st.title("🛡️ FinGuard")
st.subheader("Indian Banking Transaction Fraud Detection")

st.write(
    "Upload banking transaction data to identify potentially "
    "fraudulent transactions using a trained machine learning model."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Check the outputs folder.")
    st.stop()

if not SUMMARY_PATH.exists() or not THRESHOLD_PATH.exists():
    st.error("Model metadata or threshold file is missing.")
    st.stop()

try:
    model, summary, threshold = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# --------------------------------------------------
# 3. MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Selected Model", summary["selected_model"])
col2.metric("Decision Threshold", f"{threshold:.3f}")
col3.metric("Training Records", f'{summary["training_records"]:,}')

st.divider()


# --------------------------------------------------
# 4. FEATURE ENGINEERING FOR UPLOADED DATA
# --------------------------------------------------

def prepare_features(data, expected_features):

    data = data.copy()

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Generate date-based features if required
    if "transaction_date" in data.columns:

        dates = pd.to_datetime(
            data["transaction_date"],
            errors="coerce"
        )

        date_features = {
            "transaction_year": dates.dt.year,
            "transaction_month": dates.dt.month,
            "transaction_day": dates.dt.day,
            "transaction_dayofweek": dates.dt.dayofweek,
            "is_weekend": (
                dates.dt.dayofweek >= 5
            ).astype(float)
        }

        for name, values in date_features.items():
            if name in expected_features:
                data[name] = values

    # Align incoming data with training features
    for column in expected_features:
        if column not in data.columns:
            data[column] = np.nan

    return data[expected_features]


# --------------------------------------------------
# 5. CSV UPLOAD
# --------------------------------------------------

st.header("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload a banking transactions CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.write("Dataset preview")
        st.dataframe(uploaded_data.head())

        st.write(
            f"Total transactions: {len(uploaded_data):,}"
        )

        if st.button("Analyze Transactions", type="primary"):

            expected_features = summary["features_used"]

            features = prepare_features(
                uploaded_data,
                expected_features
            )

            probabilities = model.predict_proba(features)[:, 1]

            predictions = (
                probabilities >= threshold
            ).astype(int)

            results = uploaded_data.copy()

            results["fraud_probability"] = probabilities

            results["prediction"] = np.where(
                predictions == 1,
                "Fraud",
                "Legitimate"
            )

            results["risk_level"] = np.select(
                [
                    probabilities >= 0.75,
                    probabilities >= threshold
                ],
                [
                    "High Risk",
                    "Medium Risk"
                ],
                default="Low Risk"
            )

            st.divider()
            st.header("Fraud Detection Results")

            total = len(results)
            fraud_count = int(predictions.sum())
            legitimate_count = total - fraud_count

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Transactions", f"{total:,}")
            c2.metric("Flagged as Fraud", f"{fraud_count:,}")
            c3.metric("Legitimate", f"{legitimate_count:,}")

            st.subheader("Prediction Summary")

            st.bar_chart(
                results["prediction"].value_counts()
            )

            st.subheader("Transaction Predictions")

            st.dataframe(
                results,
                use_container_width=True
            )

            st.download_button(
                "Download Prediction Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="finguard_predictions.csv",
                mime="text/csv"
            )

            st.info(
                "Predictions are model-generated risk estimates, "
                "not confirmed cases of financial fraud. "
                "This application is a demonstration using "
                "a synthetic-labelled dataset."
            )

    except Exception as error:
        st.error(f"Error processing the dataset: {error}")


# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FinGuard | Machine Learning Project | "
    "Banking Transaction Fraud Detection"
)
