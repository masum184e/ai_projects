import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import pickle

nltk.download('punkt_tab')
nltk.download('stopwords')

stemmer=PorterStemmer()

with open("./data/coffeshop_chat.json", 'r') as f:
    intents=json.load(f)

words=[]
tags=[]
tokens_tag=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word=word_tokenize(pattern)
        words.extend(word)

        tokens_tag.append((word, tag))

ignore_words=['?','!','.',',','\'s']
words=[stemmer.stem(word) for word in words if word not in ignore_words]

words=sorted(set(words))
tags=sorted(set(tags))

def bag_of_words(tokens, words):
    stemmed_token=[stemmer.stem(token) for token in tokens]

    bag=np.zeros(len(words), dtype=np.float32)

    for idx, word in enumerate(words):
        if word in stemmed_token:
            bag[idx]=1.0
        
    return bag

X_train=[]
y_train=[]
for (tokens, tag) in tokens_tag:
    bow=bag_of_words(tokens, words)
    X_train.append(bow)

    y_train.append(tags.index(tag))

X_train=np.array(X_train)
y_train=np.array(y_train)

model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(tags), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)
model.save("chatbot_model.keras")

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('tags.pkl', 'wb') as f:
    pickle.dump(tags, f)

print("Model training complete!")