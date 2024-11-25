import streamlit as st
import pickle
import math
import requests
from bs4 import BeautifulSoup
import re
import string
import pandas as pd

# Load the pre-trained models and vectorizer filenames
model_files = {
    "SVM": "svm_model.pickle",
    "SVM_h": "svm_model_h.pickle",
    "Logistic Regression": "LR.pickle",
    "Logistic Regression_h": "LR_h.pickle",
    "Random Forest": "random_forest.pickle",
    "Random Forest_h": "random_forest_h.pickle",
    "Decision Tree": "decision_tree.pickle",
    "Decision Tree_h": "decision_tree_h.pickle",
    "KNN": "KNN.pickle",
    "KNN_h": "KNN_h.pickle",
    "Multinomial Naive Bayes": "MNB.pickle",
    "Multinomial Naive Bayes_h": "MNB_h.pickle",
}

vectorizers = {
    "TF-IDF": "tfidf_vectorizer.pickle",
    "Hashing": "hashing_vectorizer.pickle"
}

# Load models and vectorizers
loaded_models = {name: pickle.load(open(filename, 'rb')) for name, filename in model_files.items()}
loaded_vectorizers = {name: pickle.load(open(filename, 'rb')) for name, filename in vectorizers.items()}

# Function to clean and preprocess text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Function to scrape content from a URL
def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #3f51b5;
        text-align: center;
        padding-bottom: 20px;
    }
    .header {
        font-size: 24px;
        color: #673ab7;
        margin-bottom: 20px;
    }
    .background {
        background-color: #f4f7fb;
        padding: 30px;
        border-radius: 10px;
    }
    .info {
        color: #607d8b;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title with styling
st.markdown('<h1 class="title">FAKE-O-METER</h1>', unsafe_allow_html=True)

st.markdown("""
    <div class="background">
        <p class="header">Welcome to the Fake News Detector App! 🧐</p>
        <p class="info">
        This app uses 6 machine learning models to detect fake news based on the content of a provided URL.
        Simply paste the URL of the article you'd like to analyze below, and the models will predict whether it's fake or real.
        </p>
    </div>
    """, unsafe_allow_html=True)

# User input for URL
url_input = st.text_input("Enter the URL of the article:")

if url_input:
    # Scrape and preprocess content
    scraped_text = scrape_content(url_input)
    cleaned_text = wordopt(scraped_text)

    # Transform the cleaned text using both vectorizers
    tfidf_transformed = loaded_vectorizers["TF-IDF"].transform([cleaned_text])
    hashing_transformed = loaded_vectorizers["Hashing"].transform([cleaned_text])

    # Prepare results
    results = []

    for model_name, model in loaded_models.items():
        vectorizer_type = "Hashing" if "_h" in model_name else "TF-IDF"
        transformed_text = hashing_transformed if vectorizer_type == "Hashing" else tfidf_transformed
        
        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(transformed_text)[0][1]
            probability = math.ceil(pred_prob * 100)
            label = "Real News ✅" if probability > 50 else "Fake News 🚨"
        else:
            prediction = model.predict(transformed_text)[0]
            probability = 100 if prediction == 1 else 0
            label = "Real News ✅" if prediction == 1 else "Fake News 🚨"
        
        results.append([model_name.replace("_h", ""), vectorizer_type, probability, label])
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=["Model", "Vectorizer", "Probability (%)", "Label"])

    # Display the predictions
    st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
    st.dataframe(df)

