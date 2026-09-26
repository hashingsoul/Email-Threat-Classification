import streamlit as st
import joblib
import time
from pathlib import Path

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Email Threat Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }

    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }

    .report-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .threat-card {
        background-color: #fff3cd;
        border-left: 5px solid #dc3545;
    }

    .routine-card {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ System Status")

    st.markdown("### Email Threat Assessment")

    st.info(
        "This application uses a trained Logistic Regression "
        "model with TF-IDF vectorization to classify email content."
    )

    st.markdown("---")

    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Feature extraction:** TF-IDF")
    st.markdown("**Engine Version:** 1.0.0")
    st.markdown("**Status:** Online 🟢")

# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_components():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:
    model, vectorizer = load_components()

except FileNotFoundError:
    st.error("⚠️ Model files could not be found.")

    st.write("Expected files:")

    st.code(
        f"""
{MODEL_PATH}
{VECTORIZER_PATH}
"""
    )

    st.info(
        "Make sure model.pkl and vectorizer.pkl are inside the "
        "model/ folder in your GitHub repository."
    )

    st.stop()

except Exception as e:
    st.error("⚠️ Failed to load the ML model.")

    st.code(str(e))

    st.stop()

# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("🛡️ Email Threat Classifier")

st.markdown(
    """
    Analyze an email and classify it as **Routine** or
    **Potentially Threatening** using the trained machine-learning model.
    """
)

st.markdown("---")

# ============================================================
# INPUT / OUTPUT COLUMNS
# ============================================================

col1, col2 = st.columns([1.5, 1], gap="large")

# ============================================================
# INPUT
# ============================================================

with col1:

    st.markdown("### 📥 Email Analysis")

    email_text = st.text_area(
        "Email Content",
        height=300,
        placeholder=(
            "Dear Customer,\n\n"
            "Your account has been suspended. "
            "Please click the link below to verify your identity..."
        )
    )

    analyze_btn = st.button(
        "🔍 Run Threat Assessment",
        type="primary"
    )

# ============================================================
# OUTPUT
# ============================================================

with col2:

    st.markdown("### 📊 Assessment Report")

    if analyze_btn:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not email_text.strip():

            st.warning(
                "Please provide email content for analysis."
            )

        else:

            with st.spinner("Analyzing email..."):

                time.sleep(0.3)

                # ------------------------------------------------
                # TF-IDF TRANSFORMATION
                # ------------------------------------------------

                input_features = vectorizer.transform(
                    [email_text]
                )

                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                prediction = model.predict(
                    input_features
                )[0]

                # ------------------------------------------------
                # PROBABILITY
                # ------------------------------------------------

                confidence = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        input_features
                    )[0]

                    confidence = max(probabilities) * 100

                # ------------------------------------------------
                # CLASS INTERPRETATION
                # ------------------------------------------------

                # Your trained model uses:
                # 0 = Routine
                # 1 = Threat
                #
                # Change this mapping if your dataset/model
                # uses the opposite label meaning.

                try:
                    prediction_int = int(prediction)

                except (ValueError, TypeError):
                    prediction_int = prediction

                is_threat = prediction_int == 1

                # ------------------------------------------------
                # THREAT RESULT
                # ------------------------------------------------

                if is_threat:

                    confidence_text = (
                        f"{confidence:.2f}%"
                        if confidence is not None
                        else "N/A"
                    )

                    st.markdown(
                        f"""
                        <div class="report-card threat-card">

                        <h2 style="color:#dc3545;">
                        🚨 POTENTIALLY THREATENING
                        </h2>

                        <p>
                        <strong>Classification:</strong>
                        Potentially Threatening
                        </p>

                        <p>
                        <strong>Model Confidence:</strong>
                        {confidence_text}
                        </p>

                        <hr>

                        <p>
                        ⚠️ Treat this email with caution.
                        Avoid interacting with suspicious links
                        or attachments.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # ROUTINE RESULT
                # ------------------------------------------------

                else:

                    confidence_text = (
                        f"{confidence:.2f}%"
                        if confidence is not None
                        else "N/A"
                    )

                    st.markdown(
                        f"""
                        <div class="report-card routine-card">

                        <h2 style="color:#198754;">
                        ✅ ROUTINE EMAIL
                        </h2>

                        <p>
                        <strong>Classification:</strong>
                        Routine
                        </p>

                        <p>
                        <strong>Model Confidence:</strong>
                        {confidence_text}
                        </p>

                        <hr>

                        <p>
                        No threat was detected by the trained
                        classification model.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

                st.markdown("---")

                m1, m2 = st.columns(2)

                with m1:

                    st.metric(
                        "Message Length",
                        f"{len(email_text):,} chars"
                    )

                with m2:

                    st.metric(
                        "TF-IDF Features",
                        f"{input_features.nnz:,}"
                    )

    else:

        st.info(
            "Paste an email above and click "
            "**Run Threat Assessment**."
        )
