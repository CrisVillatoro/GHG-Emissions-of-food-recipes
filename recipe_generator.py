import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import emoji
import pickle

import streamlit as st

with open('/Users/CristaVillatoro/Desktop/tahini-tensor-student-code 2/FINAL-project-Cookwise/recipes_generator/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

model = keras.models.load_model('recipe_generation_rnn.h5')
model.compile()


# Stop word
STOP_WORD_TITLE = emoji.emojize(':books:')


def generate_text(model, start_string, num_generate = 1000, temperature=1.0):
    # Evaluation step (generating text using the learned model)
    
    padded_start_string = STOP_WORD_TITLE + start_string

    # Converting our start string to numbers (vectorizing).
    input_indices = np.array(tokenizer.texts_to_sequences([padded_start_string]))

    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)
        
        next_character = tokenizer.sequences_to_texts(input_indices.numpy())[0]

        text_generated.append(next_character)

    return (padded_start_string + ''.join(text_generated))

def generate_combinations(model, input_letters):
    recipe_length = 1000
    try_letters = ['', '\n', 'A', 'B', 'C', input_letters[0],input_letters[1]]
    try_temperature = [1.0, 0.8]

    for letter in try_letters:
        for temperature in try_temperature:
            generated_text = generate_text(
                model,
                start_string=letter,
                num_generate = recipe_length,
                temperature=temperature
            )
            # print(f'Attempt: "{letter}" + {temperature}')
            # print('-----------------------------------')
            # print(generated_text)
            # print('\n\n')
    
    return generated_text

# if __name__ == 'main':
#     generate_combinations(model)
#generate_combinations(model)