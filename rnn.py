import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and preprocess the Shakespeare dataset
filepath = tf.keras.utils.get_file(
    'shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]  # Use a subset of the text (adjust for memory constraints)

# Create mappings for characters to indices and vice versa
characters = sorted(set(text))
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

# Prepare sequences and the corresponding next character
SEQ_LENGTH = 40
STEP_SIZE = 3

sequences = []
next_chars = []
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sequences.append(text[i: i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])

# Convert sequences into numerical format
x = np.zeros((len(sequences), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(sequences), len(characters)), dtype=np.float32)

for i, sequence in enumerate(sequences):
    x[i] = [char_to_index[char] for char in sequence]
    y[i, char_to_index[next_chars[i]]] = 1

# Define the model
model = Sequential([
    Embedding(input_dim=len(characters), output_dim=64, input_length=SEQ_LENGTH),
    LSTM(128, return_sequences=False),
    Dense(len(characters), activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.01),
    metrics=['accuracy']
)

# Train the model with checkpointing
checkpoint = ModelCheckpoint('text_generator_best.keras', monitor='loss', save_best_only=True)
model.fit(x, y, batch_size=256, epochs=10, callbacks=[checkpoint])


# Load the best model (after training)
model = tf.keras.models.load_model('text_generator_best.keras')

# Define the sampling function
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature  # Avoid log(0)
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Text generation function
def generate_text(length, temperature=1.0):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    seed_text = text[start_index: start_index + SEQ_LENGTH]
    generated_text = seed_text

    for _ in range(length):
        input_seq = np.array([char_to_index[char] for char in seed_text]).reshape(1, -1)
        predictions = model.predict(input_seq, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated_text += next_char
        seed_text = seed_text[1:] + next_char

    return generated_text

# Generate and print text with different temperatures
for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f"--- Generated Text (Temperature: {temp}) ---")
    print(generate_text(300, temperature=temp))
    print("\n")






# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.layers import Activation, Dense, LSTM

# # Load the Shakespeare dataset
# filepath = tf.keras.utils.get_file(
#     'shakespeare.txt', 
#     'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
# )

# # Read and preprocess the text
# text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
# text = text[300000:800000]  # Using a subset of the text

# # Create mappings of characters to indices and vice versa
# characters = sorted(set(text))
# char_to_index = {c: i for i, c in enumerate(characters)}
# index_to_char = {i: c for i, c in enumerate(characters)}

# # Prepare sequences and the next character
# SEQ_LENGTH = 40
# STEP_SIZE = 3
# '''
# sentences = []
# next_char = [] 
# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i + SEQ_LENGTH])
#     next_char.append(text[i + SEQ_LENGTH])

# # One-hot encoding
# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
# y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_to_index[char]] = 1
#     y[i, char_to_index[next_char[i]]] = 1

# # Define the model
# model = Sequential([
#     LSTM(128, input_shape=(SEQ_LENGTH, len(characters))),
#     Dense(len(characters)),
#     Activation('softmax')
# ])

# # Compile the model with updated optimizer argument
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=RMSprop(learning_rate=0.01)
# )

# # # Train the model
# # model.fit(x, y, batch_size=256, epochs=4)

# # # Save the model
# # model.save('textgenerator.keras')
# '''
# model= tf.keras.models.load_model('textgenerator.keras')
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)

# def generate_text(length, temperature):
#     start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
#     generated = ''
#     sentence = text[start_index: start_index + SEQ_LENGTH]
#     generated += sentence
#     for i in range(length):
#         x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
#         for t, char in enumerate(sentence):
#             x_predictions[0, t, char_to_index[char]] = 1

#         predictions = model.predict(x_predictions, verbose=0)[0]
#         next_index = sample(predictions,
#                                  temperature)
#         next_character = index_to_char[next_index]

#         generated += next_character
#         sentence = sentence[1:] + next_character
#     return generated

# print("..................0.2................")

# print(generate_text(300, 0.2))
# print("..................0.4.................")
# print(generate_text(300, 0.4))
# print("..................0.5.................")
# print(generate_text(300, 0.5))
# print(".................0.6.................")
# print(generate_text(300, 0.6))
# print(".................0.7................")
# print(generate_text(300, 0.7))
# print("..................1.................")

# print(generate_text(300, 0.8))

# Text generation function
# def generate_text(length, temperature=1.0):
#     start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
#     seed_text = text[start_index: start_index + SEQ_LENGTH]
#     generated_text = seed_text

#     for _ in range(length):
#         input_seq = np.array([char_to_index[char] for char in seed_text]).reshape(1, -1)
#         predictions = model.predict(input_seq, verbose=0)[0]
#         next_index = sample(predictions, temperature)
#         next_char = index_to_char[next_index]

#         generated_text += next_char
#         seed_text = seed_text[1:] + next_char

#     return generated_text

# # Generate and print text with different temperatures
# for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
#     print(f"--- Generated Text (Temperature: {temp}) ---")
#     print(generate_text(300, temperature=temp))
#     print("\n")


