import os
import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ✅ Absolute paths (works in PyCharm project root)
BASE_DIR = os.path.dirname(__file__)
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(vectorizer_path, "rb") as f:
    tfidf = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ================= UI Styling =================
st.set_page_config(page_title="📩 Spam Classifier", page_icon="📧", layout="centered")

st.markdown(
    """
    <style>
        .main {
            background-color: #f9fafc;
            padding: 20px;
            border-radius: 12px;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #4CAF50;
            font-size: 16px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #45a049, #4CAF50);
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= Streamlit UI =================
st.title("📩 Email / SMS Spam Classifier")
st.markdown("### 🔍 Instantly check if a message is **Spam** or **Not Spam**")

input_sms = st.text_area("✉️ Enter your message here:")

if st.button("🚀 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.error("🚨 **Spam Detected!** ❌")
            st.markdown("⚡ This looks like an unwanted or promotional message.")
        else:
            st.success("✅ **Not Spam!** 🎉")
            st.markdown("👍 This message looks safe.")
