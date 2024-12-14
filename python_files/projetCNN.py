import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
import os
import matplotlib.pyplot as plt

# Étape 1 : Charger vos données (images et labels)
image_size = (150, 150)  # Taille des images
batch_size = 32  # Taille des lots
epochs = 30
EARLY_STOPPING = False

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
    images, labels, test_size=0.3, random_state=42
)

train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

print(f"Classes détectées : {class_indices}")

# Étape 2 : Construire le modèle CNN 
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Étape 3 : Compiler le modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Étape 4 : Ajouter un callback pour arrêter l'entraînement si la validation n'améliore pas
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

# Étape 5 : Entraîner le modèle
history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    validation_split=0.2,
    batch_size=batch_size,
    callbacks= [es] if EARLY_STOPPING else None
)

model.save("model_from_scratch.h5")

# Visualisation des performances
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Performance du modèle construit from scratch', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, len(history.history['accuracy']) + 1))

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
