# import libraries for running model 
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# load model and tokenizer (trained from colab)
model = load_model('True_model.keras')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# prediction function
def predict(tokenizer, model, text):
    # convert input text to a sequence
    sequence = tokenizer.texts_to_sequences([text])
    if len(sequence[0]) == 0:
        return '[INVALID INPUT]'  # handle empty or invalid input

    sequence = np.array(sequence).reshape(1, -1)  # match model input shape
    prediction = np.argmax(model.predict(sequence), axis=-1)  # get predicted word index

    # map the predicted index to the word
    predicted_word = ''
    for key, value in tokenizer.word_index.items():
        if value == prediction:
            predicted_word = key
            break

    return predicted_word if predicted_word != '' else '[UNKNOWN]'

# user input loop to capture text and predict next word after running the model
while True:
    input_text = input('Enter your text: ')
    if input_text.lower() == 'exit':
        break
    else:
        print('predicted next word:', predict(tokenizer, model, text=input_text))
