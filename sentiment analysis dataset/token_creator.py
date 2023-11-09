import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

# Suggested max value for our dataset is 22.773
# for info check MAX VOCAB_SIZE header in sentiment analysis "dataset/cleaned_ds/dataset_cleaner.py"
VOCAB_SIZE = 15_000
TOKENIZER_PATH = "sentiment analysis dataset/tokenizer.pkl"

df = pd.read_csv("sentiment analysis dataset/train.csv", encoding="ISO-8859-1")
df.dropna(subset=["selected_text"], inplace=True)

selected_text = []
for _, row in df.iterrows():
    text = row["selected_text"]
    selected_text.append(text)

selected_text = np.array(selected_text)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(selected_text)

with open(TOKENIZER_PATH, "wb") as writer:
    pickle.dump(tokenizer, writer)
