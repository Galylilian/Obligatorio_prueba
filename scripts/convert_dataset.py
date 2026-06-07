import os
import shutil
import zipfile

# =============================
# PATHS
# =============================
BASE_PATH = "data/raw"
OUTPUT_PATH = "data/processed"

splits = ["train", "valid", "test"]
datasets = ["ds1", "ds2"]

# =============================
# EXTRAER ZIP (SI EXISTEN)
# =============================
def extract_zip_if_needed(dataset_path):
    for file in os.listdir(dataset_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(dataset_path, file)
            print(f"📦 Intentando extraer {zip_path}")

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(dataset_path)

                os.remove(zip_path)
                print(f"✅ Extraído {file}")

            except zipfile.BadZipFile:
                print(f"❌ {file} no es un zip válido → eliminando")
                os.remove(zip_path)


# =============================
# ARREGLAR ESTRUCTURA ANIDADA
# =============================
def fix_nested_structure(dataset_path):
    contents = os.listdir(dataset_path)

    # si hay una sola carpeta → probablemente está anidado
    if len(contents) == 1:
        inner_path = os.path.join(dataset_path, contents[0])

        if os.path.isdir(inner_path):
            print(f"📂 Corrigiendo estructura anidada en {dataset_path}")

            for item in os.listdir(inner_path):
                src = os.path.join(inner_path, item)
                dst = os.path.join(dataset_path, item)
                shutil.move(src, dst)

            os.rmdir(inner_path)


# =============================
# LIMPIAR OUTPUT
# =============================
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

# =============================
# PREPARAR DATASETS
# =============================
for dataset in datasets:
    dataset_path = os.path.join(BASE_PATH, dataset)

    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset no encontrado: {dataset_path}")
        continue

    extract_zip_if_needed(dataset_path)
    fix_nested_structure(dataset_path)


# =============================
# PROCESAMIENTO (CLASIFICACIÓN CORRECTO)
# =============================
for split in splits:

    print(f"\n📂 Procesando {split}")

    fall_dir_out = os.path.join(OUTPUT_PATH, split, "fall")
    no_fall_dir_out = os.path.join(OUTPUT_PATH, split, "no_fall")

    os.makedirs(fall_dir_out, exist_ok=True)
    os.makedirs(no_fall_dir_out, exist_ok=True)

    for dataset in datasets:

        split_path = os.path.join(BASE_PATH, dataset, split)

        if not os.path.exists(split_path):
            print(f"⚠️ No existe split: {split_path}")
            continue

        labels = os.listdir(split_path)

        print(f"🔍 Labels detectados en {split_path}: {labels}")

        for label in labels:

            src_dir = os.path.join(split_path, label)

            if not os.path.isdir(src_dir):
                continue

            # 🔥 CORRECCIÓN CLAVE
            if label.lower() == "fall":
                dest_dir = fall_dir_out
            else:
                dest_dir = no_fall_dir_out

            files = os.listdir(src_dir)

            print(f"  📁 {label}: {len(files)} archivos")

            for file in files:

                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                src = os.path.join(src_dir, file)
                dst = os.path.join(dest_dir, f"{dataset}_{file}")

                shutil.copy(src, dst)

    print(f"✅ {split} listo")

print("\n✅ Dataset fusionado correctamente en data/processed")