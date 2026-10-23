import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from .const import SIZE_IMAGE

def afficher_images_echntillon(images, labels):
    classes = np.unique(labels)
    _, axes = plt.subplots(len(classes), 3)

    for i, classe in enumerate(classes):
        images_index = np.where(labels == classe)[0][:3]
        for j, index in enumerate(images_index):
            # loead et converter BGR to RGB
            image = cv2.cvtColor(cv2.imread(images[index]), cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(image)
            axes[i, j].set_title(classe)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def afficher_distribution_classes(labels):
    classes, counts = np.unique(labels, return_counts=True)
    plt.figure()
    sns.barplot(x=classes, y=counts)
    plt.title('Distribution des images')
    plt.xlabel('Classes')
    plt.show()


def afficher_images_augmentation(image_path, data_augmentation):
    # Loader l'image avec OpenCV et converter à RGB
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erreur de chargement : {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (SIZE_IMAGE, SIZE_IMAGE))

    # 2. Ajouter batch dimension pour keras (1, H, W, C)
    image_batch = tf.cast(image, tf.float32)
    image_batch = tf.expand_dims(image_batch, axis=0)

    # Appliquer l'augmentation
    image_augmentee_batch = data_augmentation(image_batch, training=True)

    # Supprimer dimension et converter à uint8 pour afficher avec Matplotlib
    image_augmentee = tf.squeeze(image_augmentee_batch, axis=0)
    image_augmentee = tf.clip_by_value(image_augmentee, 0, 255)
    image_augmentee = image_augmentee.numpy().astype("uint8")

    # Affichage
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image Normale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_augmentee)
    plt.title("Image Augmentée")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def afficher_courbes_apprentissage(history):
    # Accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Accuracy - Train")
    plt.plot(history["val_accuracy"], label="Accuracy - Validation")

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Loss - Train")
    plt.plot(history["val_loss"], label="Loss - Validation")

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()