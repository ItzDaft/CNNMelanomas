import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# 1. CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RAW_DIR = os.path.join(BASE_DIR, "data", "processed", "test_ph2")
TEST_CLEAN_DIR = os.path.join(BASE_DIR, "data", "processed_clean", "test_ph2")

MODELO_BASE = 'baseline_resnet50.h5'
MODELO_OPT = 'optimizado_resnet50.h5'

IMG_SIZE = (224, 224)

# 2. GENERADORES PARA EVALUACIÓN
# Es CRITICO usar shuffle=False para que las predicciones coincidan con las etiquetas reales
test_datagen = ImageDataGenerator(rescale=1./255)

print("Cargando imágenes de prueba (PH2 Crudo)...")
generador_crudo = test_datagen.flow_from_directory(
    TEST_RAW_DIR, target_size=IMG_SIZE, batch_size=32, class_mode='binary', shuffle=False
)

print("Cargando imágenes de prueba (PH2 Limpio)...")
generador_limpio = test_datagen.flow_from_directory(
    TEST_CLEAN_DIR, target_size=IMG_SIZE, batch_size=32, class_mode='binary', shuffle=False
)

etiquetas_reales = generador_crudo.classes # Las etiquetas son iguales para ambos

# 3. FUNCIÓN PARA EVALUAR Y GENERAR MATRIZ
def evaluar_y_graficar(ruta_modelo, generador, titulo, ax):
    if not os.path.exists(ruta_modelo):
        print(f"No se encontró el modelo {ruta_modelo}")
        return
    
    print(f"Evaluando {titulo}...")
    modelo = load_model(ruta_modelo)
    
    # Hacer predicciones
    predicciones_prob = modelo.predict(generador, verbose=1)
    # Convertir probabilidad a clase (0 o 1) usando umbral de 0.5
    predicciones_clases = (predicciones_prob > 0.5).astype(int).flatten()
    
    # Calcular matriz de confusión
    cm = confusion_matrix(etiquetas_reales, predicciones_clases)
    
    # Graficar con Seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Benigno', 'Melanoma'], 
                yticklabels=['Benigno', 'Melanoma'],
                annot_kws={"size": 14})
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicción del Modelo', fontsize=12)
    ax.set_ylabel('Diagnóstico Real (Biopsia)', fontsize=12)

# 4. CREAR LA COMPARATIVA VISUAL
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

evaluar_y_graficar(MODELO_BASE, generador_crudo, 'Modelo Crudo (Baseline)', axes[0])
evaluar_y_graficar(MODELO_OPT, generador_limpio, 'Modelo Optimizado (DullRazor+CLAHE)', axes[1])

plt.tight_layout()
plt.savefig('comparativa_matrices.png', dpi=300)
print("\n¡Evaluación terminada! Revisa el archivo 'comparativa_matrices.png'.")
plt.show()