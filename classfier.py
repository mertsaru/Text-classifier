import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = "trained_model.h5"

# Encoder
imdb_word_index = imdb.get_word_index()
def encoder(text):
    words = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [imdb_word_index[word] if word in imdb_word_index else 0 for word in words] # tokenize words that exist in our index
    return pad_sequences([tokens], maxlen=100, padding="post", value=10**3 - 1)

try:
    model = load_model(model_path)
except FileNotFoundError:
    print("No model found, train a model first")
    exit(1)
else:
    while True:
        print("Enter a review or type 'exit' to quit")
        input_text = input("> ")
        if input_text.lower() == "exit":
            break
        
        encoded_text = encoder(input_text)
        prediction = model.predict(encoded_text)
        if prediction[0] >= 0.5:
            print("Positive")
        else:
            print("Negative")