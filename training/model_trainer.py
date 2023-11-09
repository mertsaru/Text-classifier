import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D


# Parameters
MODEL_NAME = "training/trained_model.keras"
MAX_LEN = 100  # depends on the dataset, it is the lenght of the each sample
VOCAB_SIZE = 15_000
EPOCH = 10

# Load datasets
train_x = np.load("sentiment analysis dataset/cleaned_ds/train_features.npy")
train_y = np.load("sentiment analysis dataset/cleaned_ds/train_labels.npy")
test_x = np.load("sentiment analysis dataset/cleaned_ds/test_features.npy")
test_y = np.load("sentiment analysis dataset/cleaned_ds/test_labels.npy")

# Create and save the model
model = Sequential(
    [
        Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
# Train model
model.fit(train_x, train_y, epochs=EPOCH, batch_size=16, validation_split=0.2)

# Accuracy
acc = model.evaluate(test_x, test_y)[1]
print(f"Accuracy: {acc*100:.2f}")

model.save(MODEL_NAME)
