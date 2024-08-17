import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import json

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