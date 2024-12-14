import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
import os
import matplotlib.pyplot as plt

# Étape 1 : Charger vos données (images et labels)
# Remplacez 'your_dataset_loader()' par votre méthode de chargement d'images et labels
# Paramètres
image_size = (150, 150)  # Taille des images (à adapter à VGG-16)
batch_size = 32  # Taille des lots
epochs = 10
# Chargement et traitement des données
path = kagglehub.dataset_download("ibrahimserouis99/one-piece-image-classifier")
data_dir = os.path.join(path, "Data", "Data")
print("Path to dataset files:", data_dir)

# Charger les images et labels depuis le répertoire
def load_images_labels(data_dir, image_size):
    images = []
    labels = []
    class_names = os.listdir(data_dir)  # Liste des noms des classes (ex: ['Nami', 'Luffy'])
    class_indices = {name: idx for idx, name in enumerate(class_names)}  # Associer une classe à un index

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            image = tf.keras.preprocessing.image.load_img(filepath, target_size=image_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(class_indices[class_name])
    
    images = np.array(images) / 255.0  # Normaliser les pixels
    labels = np.array(labels)
    return images, labels, class_indices

# Charger les données
images, labels, class_indices = load_images_labels(data_dir, image_size)
num_classes=len(class_indices)

# Diviser en ensembles d'entraînement et de test
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42
)

# Convertir les labels en one-hot encoding
train_labels = to_categorical(train_labels, num_classes=len(class_indices))
test_labels = to_categorical(test_labels, num_classes=len(class_indices))

print(f"Classes détectées : {class_indices}")

# # Étape 2 : Prétraitement des données
# # Redimensionner les images à la taille attendue par VGG-16
# image_size = (150, 150)
# images = tf.image.resize(images, image_size)
# images = preprocess_input(images)  # Prétraitement adapté à VGG-16

# # Convertir les labels en format catégorique (one-hot encoding)
# labels = to_categorical(labels, num_classes=num_classes)

# # Diviser les données en ensembles d'entraînement et de test
# train_images, test_images, train_labels, test_labels = train_test_split(
#     images, labels, test_size=0.3, random_state=42
# )

# Étape 3 : Charger le modèle VGG-16 pré-entraîné
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Geler les couches pré-entraînées

# Étape 4 : Construire le modèle
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(512, activation='relu')  # Ajustez la taille des couches selon vos données
dense_layer_2 = layers.Dense(512, activation='relu')
prediction_layer = layers.Dense(num_classes, activation='softmax')  # Nombre de classes en sortie

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    Dropout(0.3), # Ajout de dropout pour éviter le surapprentissage
    dense_layer_2,
    Dropout(0.3), # Ajout de dropout pour éviter le surapprentissage
    prediction_layer
])

# Étape 5 : Compiler le modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Étape 6 : Ajouter un callback pour arrêter l'entraînement si la validation n'améliore pas
# es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

# Étape 7 : Entraîner le modèle
history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    validation_split=0.2,
    batch_size=32
)

model.save("modelTL1.h5")

# Visualisation des performances
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Performance du modèle avec Transfer Learning', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, epochs + 1))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs + 1, 5))
ax1.set_ylabel('Valeur d\'Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs + 1, 5))
ax2.set_ylabel('Valeur de Loss')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
ax2.legend(loc="best")

plt.show()
