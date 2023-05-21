import pandas as pd
import numpy as np
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

train_data = pd.read_csv('https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv', sep='\t', header=None)
test_data = pd.read_csv('https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv', sep='\t', header=None)

x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

train_labels = train_data.pop(0)
test_labels = test_data.pop(0)




# binaire 0 ou 1
y_train = (y_train + 1) / 2
y_test = (y_test + 1) /2


plt.plot(x_test[0])
plt.plot(x_train[0])
plt.title('Exemples de données')
plt.legend(['Première donnée de test','Première donnée d\'entraînement'])
#plt.show()

model_cnn = Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(x_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu', input_shape=(x_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = model_cnn.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

model_rnn = Sequential([
    layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_rnn = model_rnn.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))


plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.plot(history_rnn.history['accuracy'])
plt.plot(history_rnn.history['val_accuracy'])
plt.title('Précision des modèles CNN et RNN')
plt.ylabel('Précision')
plt.xlabel('Epoch')
plt.legend(['CNN - Entraînement', 'CNN - Validation', 'RNN - Entraînement', 'RNN - Validation'], loc='lower right')
#plt.show()

plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.plot(history_rnn.history['loss'])
plt.plot(history_rnn.history['val_loss'])
plt.title('Perte des modèles CNN et RNN')
plt.ylabel('Perte')
plt.xlabel('Epoch')
plt.legend(['CNN - Entraînement', 'CNN - Validation', 'RNN - Entraînement', 'RNN - Validation'], loc='upper right')
#plt.show()

test = 4
predictions_cnn = model_cnn.predict(np.expand_dims(x_test[test], axis=0))
print(f"Véritable valeur: {y_test[test]} - Prédiction: {0 if predictions_cnn[0]<0.5 else 1}")


model_cnn.save('../models/model_cnn.h5')
model_rnn.save('../models/model_rnn.h5')
