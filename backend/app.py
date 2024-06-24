from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import nltk
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
import random
import json

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


class NeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetModel, self).__init__()
        self.linearlayer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linearlayer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linearlayer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linearlayer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linearlayer2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.linearlayer3(out)
        return out


app = Flask(__name__)
CORS(app)

FILE = "DATA.pth"
intents = pd.read_json(
    "https://raw.githubusercontent.com/gopikasr/Chat-Bot---Financial-digital-Assistant/main/DataChatBot.json")
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetModel(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


def bargain(value):
    numbers = {str(i): i for i in range(1, 31)}
    options_to_reply = [
        "Sorry, this is of latest fashion, Can you raise the amount a little bit",
        "This is a very special thing, we can't give you at this much less cost",
        "Oh no sorry. Please raise a little bit"
    ]
    if value in numbers:
        if numbers[value] > 25:
            return "Yes agreed! Now, you can buy the ribbon at this price"
        else:
            return random.choice(options_to_reply)
    return "Invalid input for bargaining."


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['message']
    if sentence.isdigit() and 1 <= int(sentence) <= 30:
        response = bargain(sentence)
        return jsonify({"response": response})

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.99:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent['responses'])})
    return jsonify({"response": "I do not understand..."})


if __name__ == '__main__':
    app.run(debug=True)
