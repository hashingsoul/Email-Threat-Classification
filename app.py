import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Email Threat Classifier",
    page_icon="🛡️",
    layout="wide"
)

REPO_ID = "sagniksen/email-threat-classifier"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.pkl",
        repo_type="model"
    )

    vectorizer_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="vectorizer.pkl",
        repo_type="model"
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Unable to load the model from Hugging Face.")
    st.exception(e)
    st.stop()


st.title("🛡️ Email Threat Classifier")

st.write(
    "Enter the content of an email below to classify it as "
    "routine or potentially threatening."
)

with st.sidebar:
    st.header("Model Information")
    st.write("Model: Logistic Regression")
    st.write("Feature extraction: TF-IDF")
    st.write("Model source: Hugging Face")
    st.success("Model loaded successfully")


email_text = st.text_area(
    "Email Content",
    height=300,
    placeholder=(
        "Dear Customer,\n\n"
        "Your account has been suspended. "
        "Please click the link below to verify your identity."
    )
)

if st.button("🔍 Analyze Email", type="primary"):

    if not email_text.strip():
        st.warning("Please enter some email content first.")
    else:
        with st.spinner("Analyzing email..."):

            features = vectorizer.transform([email_text])
            prediction = model.predict(features)[0]

            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities) * 100

        if int(prediction) == 1:
            st.error("🚨 Potentially Threatening")

            st.write(
                "The model classified this email as potentially threatening."
            )

        else:
            st.success("✅ Routine Email")

            st.write(
                "The model classified this email as a routine email."
            )

        if confidence is not None:
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Message Length",
                f"{len(email_text):,} characters"
            )

        with col2:
            st.metric(
                "TF-IDF Features",
                f"{features.nnz:,}"
            )

st.caption(
    "Educational prototype for email threat classification."
)
