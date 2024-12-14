import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
import os
import matplotlib.pyplot as plt

# Étape 1 : Charger vos données (images et labels)
image_size = (224, 224)  # Taille des images (à adapter à VGG-16)
batch_size = 32  # Taille des lots
epochs = 100

# Charger les données
path = kagglehub.dataset_download("ibrahimserouis99/one-piece-image-classifier")
data_dir = os.path.join(path, "Data", "Data")
print("Path to dataset files:", data_dir)

def load_images_labels(data_dir, image_size):
    images = []
    labels = []
    class_names = os.listdir(data_dir)  # Liste des noms des classes
    class_indices = {name: idx for idx, name in enumerate(class_names)}

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

images, labels, class_indices = load_images_labels(data_dir, image_size)
num_classes = len(class_indices)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

print(f"Classes détectées : {class_indices}")

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

for layer in base_model.layers[:-20]:  # Geler tout sauf les 20 dernières couches
    layer.trainable = False

# Ajouter des couches personnalisées
modelResNet = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # Sortie pour le nombre de classes
])

modelResNet.summary()

tf.keras.backend.clear_session() # Nettoyer les anciens modèles 

# Compiler de nouveau le modèle avec un taux d'apprentissage réduit
modelResNet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continuer l'entraînement avec fine-tuning
historyResNet = modelResNet.fit(
    train_images, train_labels,
    epochs=epochs,
    validation_split=0.2,
    batch_size=32
)


modelResNet.save("./weights/modelTL_finetuned.h5")

# Visualisation des performances
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Performance du modèle avec Fine-Tuning', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, len(historyResNet.history['accuracy']) + 1))

ax1.plot(epoch_list, historyResNet.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, historyResNet.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs + 1, 5))
ax1.set_ylabel('Valeur d\'Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
ax1.legend(loc="best")

ax2.plot(epoch_list, historyResNet.history['loss'], label='Train Loss')
ax2.plot(epoch_list, historyResNet.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs + 1, 5))
ax2.set_ylabel('Valeur de Loss')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
ax2.legend(loc="best")

plt.show()