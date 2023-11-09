import os
import sys

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np


# Parameters
MODEL_PATH = "training/trained_model.keras"
MAX_LEN = 100  # depends on the dataset, it is the lenght of the each sample
VOCAB_SIZE = 15_000  # depends on the tokenizer
TOKENIZER_PATH = "sentiment analysis dataset/tokenizer.pkl"

with open(TOKENIZER_PATH, "rb") as reader:
    tokenizer = pickle.load(reader)


# Encoder
def encoder(text):
    encoded_text = np.array([text])
    encoded_text = tokenizer.texts_to_sequences(encoded_text)
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_text, maxlen=MAX_LEN, padding="post", value=VOCAB_SIZE - 1
    )
    return encoded_text


if not os.path.exists(MODEL_PATH):
    print("No model found, train a model first")
    sys.exit(1)

model = load_model(MODEL_PATH)
while True:
    print("Enter a review or type 'exit' to quit")
    input_text = input("> ")
    if input_text.lower() == "exit":
        break

    encoded_text = encoder(input_text)
    prediction = model.predict(encoded_text)
    winner_index = np.argmax(prediction[0])
    if winner_index == 0:
        print("Negative review")
    elif winner_index == 1:
        print("Neutral review")
    elif winner_index == 2:
        print("Positive review")
