import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Parameters
model_name = "trained_model.h5"
num_words = 10**3

# Dataset
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words = num_words)
train_x = sequence.pad_sequences(train_x, maxlen=100, value = num_words - 1, padding="post")
test_x = sequence.pad_sequences(test_x, maxlen=100, value = num_words - 1, padding="post")
print(train_x,"\n ",train_y)
print(type(test_x))
exit(1)
# Create model
model = Sequential([
    Embedding(10**3, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(3, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("")
# Train model
model.fit(train_x, train_y, epochs=5, batch_size=16, validation_split=0.2)

# Accuracy
acc = model.evaluate(test_x, test_y)[1]
print(f"Accuracy: {acc*100:.2f}")

model.save(model_name)