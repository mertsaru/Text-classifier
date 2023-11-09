import os

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

np.random.seed(123)

MAX_LEN = 100 # only excepts 100 words (tokens) per sample
VOCAB_SIZE = 15_000
TOKENIZER_PATH = "sentiment analysis dataset/tokenizer.pkl"

with open(TOKENIZER_PATH, "rb") as reader:
    tokenizer = pickle.load(reader)


csv_files = []
for root, dir, files in os.walk("sentiment analysis dataset"):
    for file_name in files:
        if file_name.endswith(".csv"):
            path = os.path.join(root, file_name)
            file_name = file_name.split(".")[0]
            csv_files.append([file_name, path])
        else:
            continue

for csv_file in csv_files:
    file_name, path = csv_file
    features = []
    labels = []
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.dropna(subset=["sentiment", "text"], how="any", inplace=True)
    for _, row in df.iterrows():
        text = row["text"]
        sentiment = row["sentiment"]

        # sentiment cleaning
        if sentiment == "positive":
            sentiment_value = [2]
        elif sentiment == "neutral":
            sentiment_value = [1]
        elif sentiment == "negative":
            sentiment_value = [0]
        else:
            continue
        labels.append(sentiment_value)

        # text tokenizer
        features.append(text)

    features = np.array(features)
    labels = np.array(labels, dtype=np.int64)

    features = tokenizer.texts_to_sequences(features)
    features = tf.keras.preprocessing.sequence.pad_sequences(
        features, maxlen=MAX_LEN, padding="post", value=VOCAB_SIZE - 1
    )

    # Shuffle train dataset
    if file_name == "train":
        shuffle_index = np.random.permutation(len(features))
        features = features[shuffle_index]
        labels = labels[shuffle_index]

    np.save(f"sentiment analysis dataset/cleaned_ds/{file_name}_features", features)
    np.save(f"sentiment analysis dataset/cleaned_ds/{file_name}_labels", labels)
