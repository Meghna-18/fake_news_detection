# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import re
# import string
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# import pickle
# import math
# from scrape import c

# #Load the pre-trained model and vectorizer
# model_filename = 'pickles/svm_model.pickle'
# vectorizer_filename = 'pickles/tfidf_vectorizer.pickle'
# vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
# # Train and Evaluate SVM
# model = pickle.load(open(model_filename, 'rb'))


# # Function to clean and preprocess text
# def wordopt(text):
#     # Ensure text is in lowercase
#     text = text.lower()
    
#     # Use raw strings for regular expressions to avoid escape sequence issues
#     text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
#     text = re.sub(r'\W', ' ', text)     # Replace non-word characters with spaces
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
#     text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
#     text = re.sub(r'\n', '', text)      # Remove newlines
#     text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    
#     return text

# # Input link from the user
# link = input("Enter URL: ")
# test = c(link)  # Assuming `c(link)` is the function that scrapes content from the URL

# # Clean and preprocess the scraped text
# test_x = wordopt(test)

# # Transform the cleaned text using the loaded TF-IDF vectorizer
# tfidf_x = vectorizer.transform([test_x])

# # Predict the class and calculate the probability
# pred = model.predict(tfidf_x)
# result = math.ceil(model._predict_proba_lr(tfidf_x)[0][1] * 100)  # Get the predicted probability as percentage

# # Print the result
# print(f"{result}% True.")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Ensure you are using SVC here for SVM
import pickle
import math
from scrape import c

# Load the pre-trained model and vectorizer
model_filename = 'svm_model.pickle'
vectorizer_filename = 'tfidf_vectorizer.pickle'
vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

# Load the pre-trained model
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

# Input link from the user
link = input("Enter URL: ")
test = c(link)  # Assuming `c(link)` is the function that scrapes content from the URL

# Clean and preprocess the scraped text
test_x = wordopt(test)

# Transform the cleaned text using the loaded TF-IDF vectorizer
tfidf_x = vectorizer.transform([test_x])

# Predict the class and calculate the probability
# If the model was trained with probability=True, you can use predict_proba
if hasattr(model, "predict_proba"):
    pred_prob = model.predict_proba(tfidf_x)[0][1]  # Probability for class 1
    result = math.ceil(pred_prob * 100)  # Convert to percentage
    print(f"{result}% True.")
else:
    # If the model doesn't support predict_proba (likely because it wasn't trained with probability=True)
    pred = model.predict(tfidf_x)
    print(f"Prediction: {pred}")

