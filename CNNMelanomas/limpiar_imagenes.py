import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "processed_clean")

def limpiar_imagen_medica(ruta_origen, ruta_destino):
    img = cv2.imread(ruta_origen)
    if img is None:
        return False

    # Convertir a escala de grises para crear la mascara
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crear un kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    
    # Aplicar Black-Hat para aislar los filamentos oscuros
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Crear la mascara final aislando los pixeles muy marcados
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Borrar vellos usando algoritmo TELEA
    img_sin_vello = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(img_sin_vello, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar limitador adaptativo de contraste solo al canal de luminosidad
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    cv2.imwrite(ruta_destino, final_img)
    return True

# 3. RECORRER Y LIMPIAR TODOS LOS DATASETS
carpetas_a_procesar = [
    os.path.join("train", "benigno"), os.path.join("train", "melanoma"),
    os.path.join("val", "benigno"), os.path.join("val", "melanoma"),
    os.path.join("test_ph2", "benigno"), os.path.join("test_ph2", "melanoma")
]

print("Iniciando DullRazor + CLAHE")

total_procesadas = 0

for subcarpeta in carpetas_a_procesar:
    ruta_origen_completa = os.path.join(PROCESSED_DIR, subcarpeta)
    ruta_destino_completa = os.path.join(CLEAN_DIR, subcarpeta)
    
    os.makedirs(ruta_destino_completa, exist_ok=True)
    
    if os.path.exists(ruta_origen_completa):
        archivos = os.listdir(ruta_origen_completa)
        print(f"Procesando {len(archivos)} imagenes en: {subcarpeta}")
        
        for archivo in archivos:
            origen = os.path.join(ruta_origen_completa, archivo)
            destino = os.path.join(ruta_destino_completa, archivo)
            
            # Solo procesar si es archivo de imagen
            if os.path.isfile(origen) and origen.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                exito = limpiar_imagen_medica(origen, destino)
                if exito:
                    total_procesadas += 1

print(f"\nOperacion finalizada. {total_procesadas} imagenes han sido limpiadas y guardadas en 'processed_clean'.")