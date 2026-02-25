import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# =============================
# Setup
# =============================
nltk.download('stopwords')

st.set_page_config(page_title="Fake News Detection System", layout="wide")

DATA_PATH = "dataset/user_news.csv"

# =============================
# Text Preprocessing
# =============================
def preprocess(text):

    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]

    return " ".join(words)

# =============================
# Load Dataset
# =============================
def load_dataset():

    try:
        df = pd.read_csv(DATA_PATH)

        if "News_Text" not in df.columns or "Label" not in df.columns:
            return pd.DataFrame(columns=["News_Text","Label"])

        df.dropna(inplace=True)

        return df

    except:
        return pd.DataFrame(columns=["News_Text","Label"])

# =============================
# Train Model (Stable Version ⭐)
# =============================
def train_model():

    df = load_dataset()

    if len(df) < 10:
        return None, None, 0

    df["News_Text"] = df["News_Text"].apply(preprocess)

    X = df["News_Text"]
    y = df["Label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Train-test split (Better graph quality)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, vectorizer, acc

# =============================
# Backup Live News Mode ⭐
# =============================
def fetch_live_news():

    sample_news = [
        "Government launches education scholarship program",
        "Stock market shows positive trend today",
        "New technology research project announced",
        "Health ministry starts awareness campaign",
        "University introduces new admission policy"
    ]

    try:
        api_key = "8d5dfd2bc67e45c780c4a2f9059b73ed"

        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={api_key}"

        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return sample_news

        data = response.json()

        news_list = []

        for article in data.get("articles", []):
            title = article.get("title")

            if title:
                news_list.append(title)

        return news_list if len(news_list) > 0 else sample_news

    except:
        return sample_news

# =============================
# UI DESIGN
# =============================
st.title("🧠 Fake News Detection System using Machine Learning")
st.markdown("---")

model, vectorizer, accuracy = train_model()

# =============================
# Training Data Input
# =============================
st.subheader("📌 Add Training Data")

news = st.text_area("Enter News Text")
label = st.selectbox("Select Label", ["Fake","Real"])

if st.button("Save Training Data"):

    if news.strip() != "":

        new_df = pd.DataFrame({
            "News_Text":[news],
            "Label":[label]
        })

        new_df.to_csv(DATA_PATH, mode='a', header=False, index=False)

        st.success("✅ Data Added Successfully")

# =============================
# Prediction Section
# =============================
st.markdown("---")

st.subheader("🔍 News Prediction")

test_news = st.text_area("Enter News for Detection")

if st.button("Detect News"):

    if model is None:
        st.warning("⚠ Add more training data (Minimum 10 samples)")
    else:

        processed = preprocess(test_news)

        vec = vectorizer.transform([processed])

        result = model.predict(vec)[0]

        st.success(f"Prediction → {result}")
        st.info(f"Model Accuracy → {accuracy:.2f}")

# =============================
# Accuracy Graph
# =============================
if accuracy > 0:

    st.subheader("📊 Model Accuracy Graph")

    fig = plt.figure(figsize=(4,3))

    plt.bar(["Model Accuracy"], [accuracy])

    plt.xlabel("Performance")
    plt.ylabel("Score")
    plt.ylim(0,1)

    st.pyplot(fig)

# =============================
# Live News Demo Section
# =============================
st.subheader("🌐 Live News Detection Demo")

if st.button("Fetch Live News"):

    live_news = fetch_live_news()

    for news in live_news[:5]:
        st.write("📰", news)