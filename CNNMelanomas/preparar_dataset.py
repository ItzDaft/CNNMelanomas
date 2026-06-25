import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Ajustado a los nombres exactos de tus carpetas
ISIC_RAW = os.path.join(RAW_DIR, "ISIC_2019")
HAM_RAW = os.path.join(RAW_DIR, "HAM10000") 
PH2_RAW = os.path.join(RAW_DIR, "PH2", "PH2Dataset")

TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DIR, "test_ph2")

carpetas = [
    os.path.join(TRAIN_DIR, "benigno"), os.path.join(TRAIN_DIR, "melanoma"),
    os.path.join(VAL_DIR, "benigno"), os.path.join(VAL_DIR, "melanoma"),
    os.path.join(TEST_DIR, "benigno"), os.path.join(TEST_DIR, "melanoma")
]

for carpeta in carpetas:
    os.makedirs(carpeta, exist_ok=True)

# 1. Procesar ISIC 2019: Copiar directo de la carpeta MEL
print("Procesando ISIC 2019...")
isic_mel_dir = os.path.join(ISIC_RAW, "MEL")

if os.path.exists(isic_mel_dir):
    copiadas_isic = 0
    for img_name in os.listdir(isic_mel_dir):
        origen = os.path.join(isic_mel_dir, img_name)
        destino = os.path.join(TRAIN_DIR, "melanoma", img_name)
        if os.path.isfile(origen):
            shutil.copy(origen, destino)
            copiadas_isic += 1
    print(f"ISIC: {copiadas_isic} imagenes de melanoma copiadas a Train.")
else:
    print("Error: No se encontro la carpeta MEL de ISIC.")

# 2. Procesar HAM10000: Buscar en part_1 y part_2
print("Procesando HAM10000...")
ham_csv = os.path.join(HAM_RAW, "HAM10000_metadata.csv")
part1 = os.path.join(HAM_RAW, "HAM10000_images_part_1")
part2 = os.path.join(HAM_RAW, "HAM10000_images_part_2")

df_ham = pd.read_csv(ham_csv)
df_ham['clase'] = df_ham['dx'].apply(lambda x: 'melanoma' if x == 'mel' else 'benigno')

train_ham, val_ham = train_test_split(
    df_ham, 
    test_size=0.20, 
    random_state=42, 
    stratify=df_ham['clase']
)

def copiar_ham(dataframe, carpeta_destino, nombre_set):
    copiadas = 0
    for index, row in dataframe.iterrows():
        img_name = str(row['image_id']) + ".jpg"
        clase = row['clase']
        destino = os.path.join(carpeta_destino, clase, img_name)
        
        origen_1 = os.path.join(part1, img_name)
        origen_2 = os.path.join(part2, img_name)
        
        if os.path.exists(origen_1):
            shutil.copy(origen_1, destino)
            copiadas += 1
        elif os.path.exists(origen_2):
            shutil.copy(origen_2, destino)
            copiadas += 1
    print(f"HAM10000: {copiadas} imagenes copiadas a {nombre_set}.")

copiar_ham(train_ham, TRAIN_DIR, "Train")
copiar_ham(val_ham, VAL_DIR, "Val")

# 3. Procesar PH2
print("Procesando PH2...")
ph2_csv = os.path.join(PH2_RAW, "PH2_dataset.txt")
ph2_img_base = os.path.join(PH2_RAW, "PH2 Dataset images")

def cargar_ph2(ruta_txt):
    registros = []
    with open(ruta_txt, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea.startswith("|| IMD"):
                continue
            partes = [p.strip() for p in linea.split("||")]
            if len(partes) < 4 or not partes[1].startswith("IMD"):
                continue
            try:
                clinical_dx = int(partes[3])
            except ValueError:
                continue
            registros.append({"Name": partes[1], "Clinical Diagnosis": clinical_dx})
    return pd.DataFrame(registros)

def buscar_imagen_ph2(base_dir, name):
    for raiz, directorios, archivos in os.walk(base_dir):
        for archivo in archivos:
            # Buscamos el archivo exacto sin importar en qué subcarpeta esté escondido
            if archivo == f"{name}.bmp" or archivo == f"{name}_Dermoscopic_Image.bmp":
                return os.path.join(raiz, archivo)
    return None

if os.path.exists(ph2_csv):
    df_ph2 = cargar_ph2(ph2_csv)
    copiadas_ph2 = 0
    for _, row in df_ph2.iterrows():
        name = str(row["Name"])
        clase = "melanoma" if row["Clinical Diagnosis"] == 2 else "benigno"
        origen = buscar_imagen_ph2(ph2_img_base, name)
        
        if origen is None:
            continue
            
        img_name = os.path.basename(origen)
        destino = os.path.join(TEST_DIR, clase, img_name)
        shutil.copy(origen, destino)
        copiadas_ph2 += 1
    print(f"PH2: {copiadas_ph2} imagenes copiadas de {len(df_ph2)} registros.")
else:
    print("Aviso: No se encontro archivo txt de PH2.")

print("Ejecucion finalizada. Datasets organizados.")