import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

nltk.download('punkt')       # standard tokenizer
nltk.download('punkt_tab')   # tab tokenizer
nltk.download('stopwords')   # stop words
nltk.download('wordnet')     # WordNet

BASE_PATH = os.path.join(os.getcwd(), 'models')
tfidf = pickle.load(open(os.path.join(BASE_PATH, "tfidf_vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_PATH, "model.pkl"), "rb"))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

# Preprocessing
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

# Vectorization
input_vector = tfidf.transform([transform_text(input_sms)])

# Prediction
prediction = model.predict(input_vector)[0]

# Display
if st.button("Predict"):
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

