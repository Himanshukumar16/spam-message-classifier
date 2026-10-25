import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# -------------------- App settings --------------------
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "spam_dataset.csv"
MODEL_PATH = BASE_DIR / "spam_model.pkl"
RANDOM_STATE = 42


# -------------------- Text preprocessing --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b\d{5,}\b", " phonetoken ", text)
    text = re.sub(r"[£$€]\d+[\d,]*", " moneytoken ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------- Dataset loading --------------------
def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.name}. "
            "Add spam_dataset.csv to the root of your GitHub repository."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if {"label", "message"}.issubset(df.columns):
        pass
    elif {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "label", "v2": "message"})
    elif {"category", "message"}.issubset(df.columns):
        df = df.rename(columns={"category": "label"})
    else:
        raise ValueError(
            "Could not find the label/message columns. "
            "Expected label and message, v1 and v2, or category and message."
        )

    df = df[["label", "message"]].dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    if df.empty:
        raise ValueError("The dataset contains no rows labelled 'ham' or 'spam'.")

    df["target"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    df["clean"] = df["message"].apply(clean_text)

    if df["target"].nunique() < 2:
        raise ValueError("The dataset must contain both ham and spam messages.")

    return df


# -------------------- Model creation --------------------
def build_pipeline():
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=80000,
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )

    features = FeatureUnion([
        ("char_tfidf", char_tfidf),
        ("word_tfidf", word_tfidf),
    ])

    # Calibration enables probability estimates for the UI.
    classifier = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
        cv=3,
        method="sigmoid",
    )

    return Pipeline([
        ("features", features),
        ("clf", classifier),
    ])


@st.cache_resource(show_spinner="Loading or training the spam classifier...")
def get_model():
    # Reuse an existing model if one is included in the repository.
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), "loaded"
        except Exception:
            # If the saved model is incompatible/corrupt, train from the dataset.
            pass

    df = load_dataset(DATASET_PATH)
    X = df["clean"]
    y = df["target"]

    # A small dataset can fail a stratified split; training on all rows avoids
    # unnecessary deployment failure. We only need the trained model for inference.
    model = build_pipeline()
    model.fit(X, y)

    # Saving is optional: some hosted environments may have a read-only filesystem.
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass

    return model, "trained"


# -------------------- User interface --------------------
st.title("📩 Spam Message Classifier")
st.write(
    "Enter an SMS or message below to check whether it is likely to be spam "
    "or a legitimate message."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This project uses character-level and word-level TF-IDF features "
        "with a calibrated Linear Support Vector Classifier."
    )
    st.caption("Predictions are estimates, not a guarantee that a message is safe.")

try:
    model, model_status = get_model()
    if model_status == "loaded":
        st.caption("Loaded the saved model.")
    else:
        st.caption("Model trained from the repository dataset.")
except Exception as exc:
    st.error("The classifier could not start.")
    st.code(str(exc))
    st.info(
        "Check that spam_dataset.csv is committed to the root of your GitHub "
        "repository and contains valid ham/spam labels and message text."
    )
    st.stop()

message = st.text_area(
    "Message to check",
    placeholder="Example: Congratulations! You have won a prize. Click here...",
    height=150,
)

if st.button("Check message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        try:
            prediction = int(model.predict([cleaned])[0])
            probabilities = model.predict_proba([cleaned])[0]
            classes = list(model.classes_)
            spam_probability = float(probabilities[classes.index(1)]) * 100
            ham_probability = float(probabilities[classes.index(0)]) * 100

            if prediction == 1:
                st.error("🚨 Likely SPAM")
            else:
                st.success("✅ Likely LEGITIMATE (Ham)")

            col1, col2 = st.columns(2)
            col1.metric("Spam probability", f"{spam_probability:.1f}%")
            col2.metric("Legitimate probability", f"{ham_probability:.1f}%")

            st.progress(int(round(spam_probability)))
            st.caption(
                "Probability values are model estimates and may not be calibrated "
                "perfectly for every type of message."
            )
        except Exception as exc:
            st.error("Prediction failed.")
            st.code(str(exc))

st.divider()
st.caption("Built with Python, scikit-learn and Streamlit.")
