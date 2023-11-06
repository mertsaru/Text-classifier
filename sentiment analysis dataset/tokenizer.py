import os
import csv

import numpy as np
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary

max_len = 100
my_vocab = Dictionary.load("sentiment analysis dataset/tokens")
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
    with open(path, "r") as reader:
        csv_reader = csv.reader(reader)
        for row in csv_reader:
            text, sentiment = row[1:3]
            # sentiment cleaning
            if sentiment.lower() == "positive":
                sentiment_array = np.array([4], dtype=np.int64)
            elif sentiment.lower() == "neutral":
                sentiment_array = np.array([2], dtype=np.int64)
            elif sentiment.lower() == "negative":
                sentiment_array = np.array([0], dtype=np.int64)
            else:
                continue
            try:
                labels = np.vstack((labels, sentiment_array))
            except NameError:
                labels = sentiment_array
                print("first run of the labels")

            # text tokenizer
            words = word_tokenize(text)
            tokenized_text = [
                my_vocab.token2id[word] if word in my_vocab.token2id.keys() else 0
                for word in words
            ]
            if len(tokenized_text) < max_len:
                tokenized_text = tokenized_text + [0] * (max_len - len(tokenized_text))
            else:
                tokenized_text = tokenized_text[:max_len]
            text_array = np.array(tokenized_text, dtype=np.int64)
            try:
                features = np.vstack((features, text_array))
            except NameError:
                features = text_array
                print("first run of the features")


    np.save(f"sentiment analysis dataset/cleaned_ds/{file_name}_features", features)
    np.save(f"sentiment analysis dataset/cleaned_ds/{file_name}_labels", labels)
    del features, labels
