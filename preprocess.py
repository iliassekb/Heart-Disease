import os
import shutil
import cv2
import random

# --- Dossiers source ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(BASE_PATH, "Dataset")
output_dir = os.path.join(BASE_PATH, "processed_dataset")

normal_dir = os.path.join(base_dir, "Normal Person ECG Images")
abnormal_dirs = [
    os.path.join(base_dir, "ECG Images of Myocardial Infarction Patients"),
    os.path.join(base_dir, "ECG Images of Patient that have abnormal heartbeat"),
    os.path.join(base_dir, "ECG Images of Patient that have History of MI")
]

# --- Dossiers cibles ---
normal_out = os.path.join(output_dir, "Normal")
abnormal_out = os.path.join(output_dir, "Anormal")
os.makedirs(normal_out, exist_ok=True)
os.makedirs(abnormal_out, exist_ok=True)


def clear_directory(path):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path) or os.path.islink(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)


clear_directory(normal_out)
clear_directory(abnormal_out)

# --- Paramètres ---
target_size = (224, 224)
num_normal = 284
total_anormal = 284  # nombre total d'images anormales
blur_kernel_size = (5, 5)

# Répartition équilibrée entre les 3 catégories anormales
num_anomalous_images = [round(total_anormal / 3)] * 3  # environ 95 chacune

def process_and_save_images(input_paths, output_dir, limit=None):
    random.shuffle(input_paths)
    if limit:
        input_paths = input_paths[:limit]
    for path in input_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)
            equalized = cv2.equalizeHist(blurred)
            resized = cv2.resize(equalized, target_size, interpolation=cv2.INTER_AREA)
            filename = os.path.basename(path)
            cv2.imwrite(os.path.join(output_dir, filename), resized)
        except Exception as e:
            print(f"Erreur sur {path}: {e}")

# --- Normal ---
normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
process_and_save_images(normal_images, normal_out, limit=num_normal)

# --- Anormal (équilibré à 284 au total) ---
for subdir, limit in zip(abnormal_dirs, num_anomalous_images):
    imgs = [os.path.join(subdir, f) for f in os.listdir(subdir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    process_and_save_images(imgs, abnormal_out, limit=limit)

# --- Vérification finale ---
num_final_normal = len(os.listdir(normal_out))
num_final_anormal = len(os.listdir(abnormal_out))
print(f"Dataset traité et équilibré avec succès !")
print(f"Nombre d'images normales : {num_final_normal}")
print(f"Nombre d'images anormales : {num_final_anormal}")
