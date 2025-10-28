import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# ----------------- Load Corpus -----------------
file_path = tf.keras.utils.get_file(
    'poetry.txt',  # can be any text file with poems
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(file_path, 'r', encoding='utf-8').read().lower()
print(f'Corpus length: {len(text)} characters')

# ----------------- Prepare Vocabulary -----------------
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

# ----------------- Create Sequences -----------------
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

print("Number of sequences:", len(sentences))

# ----------------- One-hot encode -----------------
VOCAB = len(characters)
x = np.zeros((len(sentences), SEQ_LENGTH, VOCAB), dtype=np.float32)
y = np.zeros((len(sentences), VOCAB), dtype=np.float32)

for i, sentence in enumerate(sentences):
    for t, ch in enumerate(sentence):
        x[i, t, char_to_index[ch]] = 1.0
    y[i, char_to_index[next_characters[i]]] = 1.0

# ----------------- Build LSTM Model -----------------
model = Sequential([
    tf.keras.Input(shape=(SEQ_LENGTH, VOCAB)),
    LSTM(128),
    Dense(VOCAB),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(0.001), metrics=['accuracy'])
model.summary()

# ----------------- Train -----------------
model.fit(x, y, batch_size=256, epochs=4, validation_split=0.1)

# ----------------- Save Model and Mappings -----------------
out_model_path = "poetry_model.keras"
model.save(out_model_path)

chars_path = "chars.json"
with open(chars_path, "w", encoding="utf-8") as f:
    json.dump({
        "char_to_index": char_to_index,
        "index_to_char": {str(k): v for k, v in index_to_char.items()}
    }, f, ensure_ascii=False, indent=2)
print("Model and mappings saved!")

# ----------------- Text Generation -----------------
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, length=300, temperature=1.0):
    seed_text = seed_text.lower()
    if len(seed_text) < SEQ_LENGTH:
        seed_text = " " * (SEQ_LENGTH - len(seed_text)) + seed_text

    generated = seed_text
    sentence = seed_text[-SEQ_LENGTH:]

    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, VOCAB), dtype=np.float32)
        for t, ch in enumerate(sentence):
            if ch in char_to_index:
                x_pred[0, t, char_to_index[ch]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# ----------------- Generate Sample Text -----------------
seed = "upon the quiet night the moon"
print("\nGenerated Poetry:\n")
print(generate_text(model, seed, length=500, temperature=0.7))
