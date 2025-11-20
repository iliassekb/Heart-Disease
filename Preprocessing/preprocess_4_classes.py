import os
import shutil
import cv2
import random
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(BASE_PATH, "../Dataset")
output_dir = os.path.join(BASE_PATH, "../processed_dataset_4_classes")

# Définir les 4 classes
class_dirs = {
    "Normal": os.path.join(base_dir, "Normal Person ECG Images"),
    "Myocardial_Infarction": os.path.join(base_dir, "ECG Images of Myocardial Infarction Patients"),
    "Abnormal_Heartbeat": os.path.join(base_dir, "ECG Images of Patient that have abnormal heartbeat"),
    "History_MI": os.path.join(base_dir, "ECG Images of Patient that have History of MI")
}

# Créer les dossiers de sortie
class_output_dirs = {}
for class_name in class_dirs.keys():
    output_path = os.path.join(output_dir, class_name)
    os.makedirs(output_path, exist_ok=True)
    class_output_dirs[class_name] = output_path

def clear_directory(path):
    if not os.path.exists(path):
        return
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path) or os.path.islink(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)

# Nettoyer les dossiers de sortie
for output_path in class_output_dirs.values():
    clear_directory(output_path)

target_size = (224, 224)
blur_kernel_size = (5, 5)
max_images_per_class = None  # None = traiter toutes les images

def auto_crop_content(img_gray_uint8: np.ndarray, pad: int = 8) -> np.ndarray:
    _, th = cv2.threshold(img_gray_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) == 0:
        return img_gray_uint8
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    H, W = img_gray_uint8.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return img_gray_uint8[y0:y1, x0:x1]

def process_and_save_images(input_paths, output_dir, limit=None):
    random.shuffle(input_paths)
    if limit:
        input_paths = input_paths[:limit]
    
    success_count = 0
    error_count = 0
    
    for idx, path in enumerate(input_paths, 1):
        try:
            img = cv2.imread(path)
            if img is None:
                error_count += 1
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)
            equalized = cv2.equalizeHist(blurred)
            cropped = auto_crop_content(equalized)
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
            
            filename = os.path.basename(path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized)
            success_count += 1
            
            if idx % 50 == 0:
                print(f"  Traité {idx}/{len(input_paths)} images...")
                
        except Exception as e:
            error_count += 1
            print(f"  Erreur sur {os.path.basename(path)}: {e}")
    
    return success_count, error_count

# Traiter toutes les classes
print("=" * 70)
print("TRAITEMENT DES IMAGES PAR CLASSE")
print("=" * 70)

results = {}

for class_name, input_dir in class_dirs.items():
    if not os.path.exists(input_dir):
        print(f"\n⚠️  Dossier introuvable pour {class_name}: {input_dir}")
        continue
    
    print(f"\n>>> Traitement de la classe: {class_name}")
    print(f"    Source: {input_dir}")
    print(f"    Destination: {class_output_dirs[class_name]}")
    
    image_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    print(f"    Images trouvées: {len(image_files)}")
    
    success, errors = process_and_save_images(
        image_files, 
        class_output_dirs[class_name], 
        limit=max_images_per_class
    )
    
    results[class_name] = {
        'total': len(image_files),
        'processed': success,
        'errors': errors
    }
    
    print(f"    ✅ Traitées avec succès: {success}")
    if errors > 0:
        print(f"    ❌ Erreurs: {errors}")

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
for class_name, stats in results.items():
    print(f"{class_name:25s}: {stats['processed']:4d} images traitées (sur {stats['total']:4d} disponibles)")
print("=" * 70)

# Vérification finale
print("\n" + "=" * 70)
print("VÉRIFICATION DES RÉSULTATS")
print("=" * 70)
for class_name, output_path in class_output_dirs.items():
    if os.path.exists(output_path):
        num_images = len([f for f in os.listdir(output_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{class_name:25s}: {num_images:4d} images dans {output_path}")
    else:
        print(f"{class_name:25s}: ❌ Dossier introuvable")

print("\n✅ Preprocessing terminé avec succès !")
print(f"📁 Images sauvegardées dans: {output_dir}")





