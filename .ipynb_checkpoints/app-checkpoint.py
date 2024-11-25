import streamlit as st
import pickle
import math
import requests
from bs4 import BeautifulSoup
import re
import string

# Load the pre-trained model and vectorizer
model_filename = 'svm_model.pickle'
vectorizer_filename = 'tfidf_vectorizer.pickle'

# Load the pre-trained vectorizer
vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

# Load the pre-trained SVM model
model = pickle.load(open(model_filename, 'rb'))

# Function to clean and preprocess text
def wordopt(text):
    # Ensure text is in lowercase
    text = text.lower()
    
    # Use raw strings for regular expressions to avoid escape sequence issues
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
    text = re.sub(r'\W', ' ', text)     # Replace non-word characters with spaces
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)      # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    
    return text

# Function to scrape content from a URL
def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all the text from the page
    text = soup.get_text()
    
    return text

# Streamlit UI for user input
st.title("Fake News Detection")

st.write("""
         This app uses a machine learning model to detect fake news based on the content from a provided URL.
         Simply paste the URL of the article you'd like to analyze below, and the model will make a prediction.
         """)

# Input link from the user
url_input = st.text_input("Enter URL:")

if url_input:
    # Scrape content from the URL
    scraped_text = scrape_content(url_input)
    
    # Clean and preprocess the scraped text
    test_x = wordopt(scraped_text)
    
    # Transform the cleaned text using the loaded TF-IDF vectorizer
    tfidf_x = vectorizer.transform([test_x])
    
    # Predict the class and calculate the probability
    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(tfidf_x)[0][1]  # Probability for class 1 (fake)
        result = math.ceil(pred_prob * 100)  # Convert to percentage
        st.write(f"Prediction: {result}% chance of being fake news.")
    else:
        pred = model.predict(tfidf_x)
        if pred[0] == 1:
            st.write("Prediction: Fake News!")
        else:
            st.write("Prediction: Real News!")
