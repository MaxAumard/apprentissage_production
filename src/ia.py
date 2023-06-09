import pandas as pd
import numpy as np
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

# Chargement des données d'entraînement
train_data = pd.read_csv('https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv', sep='\t', header=None)
# Chargement des données de test
test_data = pd.read_csv('https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv', sep='\t', header=None)

# Extraction des caractéristiques d'entraînement
x_train = train_data.iloc[:, 1:].values
# Extraction des étiquettes de classe d'entraînement
y_train = train_data.iloc[:, 0].values

# Extraction des caractéristiques de test
x_test = test_data.iloc[:, 1:].values
# Extraction des étiquettes de classe de test
y_test = test_data.iloc[:, 0].values

# Suppression et recuperation de la première colonne des données d'entraînement et de test
train_labels = train_data.pop(0)
test_labels = test_data.pop(0)

# Conversion des étiquettes en binaire (0 ou 1)
y_train = (y_train + 1) / 2
y_test = (y_test + 1) / 2

# Modèle de réseau neuronal convolutionnel (CNN)
model_cnn = Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(x_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu', input_shape=(x_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilation du modèle CNN
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Entraînement du modèle CNN sur les données d'entraînement sur 50 époques
model_cnn.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Modèle de réseau neuronal récurrent (RNN)
model_rnn = Sequential([
    layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilation du modèle RNN
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Entraînement du modèle RNN avec 50 époques
model_rnn.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Sauvegarde du modèle CNN dans un fichier h5
model_cnn.save('../models/model_cnn.h5')
model_rnn.save('../models/model_rnn.h5')
