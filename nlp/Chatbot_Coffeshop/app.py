import json
import numpy as np
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

stemmer = PorterStemmer()

model = load_model('chatbot_model.h5')

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('tags.pkl', 'rb') as f:
    tags = pickle.load(f)

with open("./data/coffeshop_chat.json", 'r') as f:
    intents = json.load(f)

def bag_of_words(user_input):
    tokens = word_tokenize(user_input)
    stemmed_token=[stemmer.stem(token) for token in tokens]

    bag=np.zeros(len(words), dtype=np.float32)

    for idx, word in enumerate(words):
        if word in stemmed_token:
            bag[idx]=1.0
        
    return bag

def predict_tag(user_input):
    bow=bag_of_words(user_input)
    res=model.predict(np.array([bow]), verbose=0)[0]
    threshold = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    if results:
        return tags[results[0][0]]
    else:
        return "no_match" 
    return res

def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

    return "I'm sorry, I don't understand that."

def chatbot_response(user_input):
    predicted_tag = predict_tag(user_input)
    if predicted_tag == "no_match":
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
    
    response = get_response(predicted_tag, intents)
    return response

print("Start chatting with the bot (type 'quit' to stop):")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    print("Bot:", chatbot_response(user_input))