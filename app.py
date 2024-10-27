import json
import random
import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# Set the NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Print NLTK data paths for verification
print("NLTK data paths:", nltk.data.path)

# Download necessary NLTK data (if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    intents = json.load(file)

# Load the model
model = load_model('chatbot_model.h5')

# Data Preprocessing
words = []
classes = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Define functions for predicting and getting responses
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Streamlit App
st.title("Mental Health Chatbot")
st.write("I'm here to support you. How can I help?")

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        ints = predict_class(user_input, model)
        response = get_response(ints, intents)
        st.write("**Bot:**", response)
    else:
        st.write("Please enter a message.")
