# Text Prediction Model

This project implements a text prediction model using TensorFlow and Keras. The model predicts the next word in a given text sequence using recurrent neural networks (RNNs).



## Recurrent Neural Network

A recurrent neural network (RNN) is a type of artificial neural network designed to handle sequential data, such as text, time-series, and audio. Unlike feedforward networks, RNNs maintain a "memory" of previous inputs, allowing them to learn patterns and dependencies across sequences.

### Key Features of an RNN:
1. **Sequential Processing:** RNNs process inputs in a sequence, making them well suited for language modeling tasks.
2. **Hidden State:** Maintains information about previous inputs through a hidden state that gets updated at each step.
3. **Shared Weights:** The same set of weights is applied across all time steps, reducing the model's complexity.
4. **Backpropagation Through Time (BPTT):** Used to update weights during training.

![RNN General Structure](/diagrams/RNN_diagram.jpg)

<center> General structure of an RNN</center>



## Project Outline

### **Step 1: Import Dataset**
- You can use any dataset to train the model, but this specific implementation is trained on `True.txt`, which is sourced from the [Fake News Dataset](https://www.kaggle.com/datasets/jainpooja/fake-news-detection) and converted to a text file for processing.

### **Step 2: Preprocess Data**

1. **Clean the Text:**
   - Remove unnecessary and unwanted characters and words (e.g., `.`, `--`, `?`).
   - Convert all text to lowercase to maintain uniformity.

2. **Tokenize the Corpus and Prepare Sequences:**
   - Create a `word_index` (vocabulary) mapping words to unique integers.
   - Convert sentences into sequences based on their `word_index`
   - Example:
     ```python
     import re
     from tensorflow.keras.preprocessing.text import Tokenizer
     import numpy as np

     # clean the text
     data = re.sub(r'[^\w\s]', '', data.strip().lower())

     # tokenize the text
     tokenizer = Tokenizer()
     tokenizer.fit_on_texts([data])
     sequence_data = tokenizer.texts_to_sequences([data])[0]

     # create sequences of 3 input words and 1 target word
     sequence = []
     for i in range(3, len(sequence_data)):
         words = sequence_data[i-3:i+1]
         sequence.append(words)

     # split into features (X) and labels (Y)
     x = np.array([seq[:3] for seq in sequence])  # first 3 words as input
     y = np.array([seq[3] for seq in sequence])  # 4th word as the label
     ```

3. **Join Text into a Single String:**
   - store this continious text within data varible.

**Output of Preprocessing:**
- Input (`x`): A 2D array of shape `(num_samples, 3)` containing tokenized sequences of 3 words.
- Labels (`y`): A 1D array of shape `(num_samples,)` containing the target word for each sequence.

---


### **Step 4:Tune the Neural Network Model**

The model used is a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The structure includes:

1. **Embedding Layer:**
   - Converts input words into dense vector representations.
   - Example: `Embedding(input_dim=vocabulary_size, output_dim=128, input_length=sequence_length)`
2. **LSTM Layers:**
   - Captures sequential dependencies.
   - Example:
     ```python
     model.add(LSTM(128, return_sequences=True))
     model.add(LSTM(128))
     ```
3. **Dense Layers:**
   - Fully connected layers to output the next word's probability distribution.
   - Example:
     ```python
     model.add(Dense(vocabulary_size, activation='softmax'))
     ```

4. **Compilation and Training:**
   - Compile the model with categorical cross-entropy loss and the Adam optimizer.
   - Train the model with a specified number of epochs and batch size(change these as needed depending on size of dataset).

---

### **Step 5: Predict the Next Word**

After training, switch to VS code and install the `requirements` file and make sure the `tokenizer.pkl` file is present in the IDE, finally the trained model predicts the next word in a sequence in the terminal.

