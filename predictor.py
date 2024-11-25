import pickle
import math
import requests
from bs4 import BeautifulSoup

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

# Input link from the user
link = input("Enter URL: ")
scraped_text = scrape_content(link)

# Clean and preprocess the scraped text
test_x = wordopt(scraped_text)

# Transform the cleaned text using the loaded TF-IDF vectorizer
tfidf_x = vectorizer.transform([test_x])

# Predict the class and calculate the probability
if hasattr(model, "predict_proba"):
    pred_prob = model.predict_proba(tfidf_x)[0][1]  # Probability for class 1
    result = math.ceil(pred_prob * 100)  # Convert to percentage
    print(f"{result}% True.")
else:
    pred = model.predict(tfidf_x)
    print(f"Prediction: {pred}")