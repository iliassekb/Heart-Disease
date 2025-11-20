import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# --- Dossier du dataset brut ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_PATH, "Dataset")

# --- Sous-dossiers ---
categories = {
    "Normal": os.path.join(dataset_dir, "Normal Person ECG Images"),
    "Myocardial Infarction": os.path.join(dataset_dir, "ECG Images of Myocardial Infarction Patients"),
    "Abnormal Heartbeat": os.path.join(dataset_dir, "ECG Images of Patient that have abnormal heartbeat"),
    "History of MI": os.path.join(dataset_dir, "ECG Images of Patient that have History of MI")
}

# --- Comptage des images ---
counts = {cat: len(os.listdir(path)) for cat, path in categories.items()}
print("Nombre d'images par catégorie (avant preprocessing) :")
for cat, count in counts.items():
    print(f"{cat}: {count}")

# --- Répartition Normal / Anormal ---
normal_count = counts["Normal"]
anormal_count = counts["Myocardial Infarction"] + counts["Abnormal Heartbeat"] + counts["History of MI"]

plt.figure(figsize=(6,4))
plt.bar(["Normal", "Anormal"], [normal_count, anormal_count], color=['green','red'])
plt.title("Répartition Normal vs Anormal")
plt.ylabel("Nombre d'images")
plt.show()

# --- Affichage d'exemples d'images ---
def show_samples(folder, title, n=5):
    files = random.sample(os.listdir(folder), min(n,len(os.listdir(folder))))
    plt.figure(figsize=(12,3))
    for i, f in enumerate(files):
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)
        plt.subplot(1,n,i+1)
        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
    plt.show()

# Exemples pour chaque catégorie
for cat, folder in categories.items():
    show_samples(folder, cat)

# --- Analyse des dimensions ---
dims = []
for cat, folder in categories.items():
    for f in os.listdir(folder):
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)
        if img is not None:
            dims.append(img.shape[:2])  # hauteur, largeur

dims = np.array(dims)
print(f"Dimensions des images (hauteur x largeur) : min={dims.min(axis=0)}, max={dims.max(axis=0)}, mean={dims.mean(axis=0).astype(int)}")

# --- Analyse simple de luminosité (niveau de gris moyen) ---
brightness = []
for cat, folder in categories.items():
    for f in os.listdir(folder):
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            brightness.append(img.mean())

plt.figure(figsize=(6,4))
plt.hist(brightness, bins=50, color='gray')
plt.title("Distribution de la luminosité moyenne des ECG")
plt.xlabel("Intensité moyenne")
plt.ylabel("Nombre d'images")
plt.show()
