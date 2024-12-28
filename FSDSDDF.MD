import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#import pandas
import pandas as pd

# charger le fichier csv
data = pd.read_csv('/content/realistic_dataset_products.csv')
data.head()



data.info()
label_encoder = LabelEncoder()
data["type_encoded"] = label_encoder.fit_transform(data["Type"])
produits = label_encoder.classes_
print(data.head())

# tokenizer les descriptions en mots
tokenize = Tokenizer()
# Tokenize descriptions
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(data["Description"])
sequences = tokenizer.texts_to_sequences(data["Description"])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
print(padded_sequences)

data["price_normalized"] = data["Price"] / data["Price"].max()
print(data.head())

# Préparer les données pour le modèle
X = np.hstack((padded_sequences, data["Price"].values.reshape(-1, 1)))
y = data["type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# definition du model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(produits), activation='softmax')
])
# compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

              history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# entrainement du model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#evaluation du model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

prix = {
    "Type": ["Laptop"],
    "Price": [200],
    "Description": ["un pc performant pour le developpement web"],
    
}
prix = {
    "Type": ["Laptop"],
    "Price": [200],
    "Description": ["un pc performant pour le developpement web"],
    
}
















import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Charger le fichier CSV
data = pd.read_csv('/content/realistic_dataset_products.csv')
data.head()

data.info()
label_encoder = LabelEncoder()
data["type_encoded"] = label_encoder.fit_transform(data["Type"])
produits = label_encoder.classes_
print(data.head())

# Tokenizer les descriptions en mots
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(data["Description"])
sequences = tokenizer.texts_to_sequences(data["Description"])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
print(padded_sequences)

data["price_normalized"] = data["Price"] / data["Price"].max()
print(data.head())

# Préparer les données pour le modèle
X = np.hstack((padded_sequences, data["price_normalized"].values.reshape(-1, 1)))
y = data["type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition du modèle
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(produits), activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Évaluation du modèle
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# Prédiction sur nouvelles données
def predict_product_type(description, price):
    # Tokenizer et padding de la description
    sequence = tokenizer.texts_to_sequences([description])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    # Normalisation du prix
    price_normalized = price / data["Price"].max()

    # Préparation des données pour la prédiction
    X_new = np.hstack((padded_sequence, np.array([[price_normalized]])))

    # Prédiction
    prediction = model.predict(X_new)
    predicted_class = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_class)

# Exemple de prédiction
prix = {
    "Type": ["Laptop"],
    "Price": [200],
    "Description": ["un pc performant pour le developpement web"],
}

predicted_type = predict_product_type(prix["Description"][0], prix["Price"][0])
print("Predicted Type:", predicted_type)
