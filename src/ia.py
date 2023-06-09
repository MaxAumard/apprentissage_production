import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
import time
# Chargement des données
train_data = pd.read_csv('https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv', sep='\t', header=None)
test_data = pd.read_csv('https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv', sep='\t', header=None)

train_labels = train_data.pop(0)
test_labels = test_data.pop(0)

# Encodage one hote
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)

# Modèle de réseau neuronal convolutionnel (CNN)
model_cnn = Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(train_data.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# Modèle de réseau neuronal récurrent (RNN)
model_rnn = Sequential([
    layers.LSTM(units=64, return_sequences=True, input_shape=(train_data.shape[1], 1)),
    layers.Dropout(0.5),
    layers.LSTM(units=64),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# Compilation
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrainement
history_cnn = model_cnn.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
history_rnn = model_rnn.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))


# Sauvegarde du modèle CNN dans un fichier h5
model_cnn.save('../models/model_cnn.h5')
model_rnn.save('../models/model_rnn.h5')
