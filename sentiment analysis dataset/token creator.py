import csv

from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary

# tokenizer
dictionary = Dictionary()


def tokenizer(text):
    tokens = word_tokenize(text)
    dictionary.add_documents([tokens])
    return dictionary


with open("sentiment analysis dataset/train.csv", "r") as reader:
    csv_reader = csv.reader(reader)
    # skip header
    next(csv_reader)
    for row in csv_reader:
        text = row[1]
        my_vocab = tokenizer(text)

print(my_vocab.token2id.keys())

#my_vocab.save("sentiment analysis dataset/tokens")
