import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Chemins ---
base_dir = "processed_dataset"
normal_dir = os.path.join(base_dir, "Normal")
abnormal_dir = os.path.join(base_dir, "Anormal")

# --- Comptage des images ---
num_normal = len(os.listdir(normal_dir))
num_abnormal = len(os.listdir(abnormal_dir))

print(f"Nombre d'images normales : {num_normal}")
print(f"Nombre d'images anormales : {num_abnormal}")

# --- Visualisation de quelques images ---
def show_samples(folder, title, n=5):
    plt.figure(figsize=(12, 3))
    files = os.listdir(folder)[:n]
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

show_samples(normal_dir, "Exemples - ECG Normaux")
show_samples(abnormal_dir, "Exemples - ECG Anormaux")

# --- Distribution des intensités ---
def plot_intensity_distribution(folder, label, sample_size=50):
    files = os.listdir(folder)[:sample_size]
    intensities = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        intensities.append(img.mean())
    return np.array(intensities)

normal_intensity = plot_intensity_distribution(normal_dir, "Normal")
abnormal_intensity = plot_intensity_distribution(abnormal_dir, "Anormal")

plt.figure(figsize=(8,5))
sns.kdeplot(normal_intensity, label="Normal", fill=True)
sns.kdeplot(abnormal_intensity, label="Anormal", fill=True, color="red")
plt.title("Distribution des intensités moyennes des ECG")
plt.xlabel("Intensité moyenne (niveau de gris)")
plt.legend()
plt.show()

# --- Taille et dimensions ---
example_img = cv2.imread(os.path.join(normal_dir, os.listdir(normal_dir)[0]), cv2.IMREAD_GRAYSCALE)
print(f"Dimensions d'une image après preprocessing : {example_img.shape}")
