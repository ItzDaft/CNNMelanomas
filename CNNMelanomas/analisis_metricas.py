import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. CARGAR LOS ARCHIVOS CSV
ruta_base = 'metricas_baseline.csv'
ruta_opt = 'metricas_optimizado.csv'

if not os.path.exists(ruta_base) or not os.path.exists(ruta_opt):
    print("Error: Asegúrate de que ambos archivos CSV estén en la carpeta principal.")
    exit()

df_base = pd.read_csv(ruta_base)
df_opt = pd.read_csv(ruta_opt)

# 2. CONFIGURAR LA FIGURA
plt.figure(figsize=(18, 6))
plt.suptitle('Comparativa de Rendimiento: ResNet50 Crudo vs. Preprocesamiento OpenCV', fontsize=16, fontweight='bold')

# 3. GRÁFICA 1: RECALL (Sensibilidad Médica)
plt.subplot(1, 3, 1)
plt.plot(df_base['epoch'], df_base['val_recall'], label='Crudo (Baseline)', color='red', linestyle='--', marker='o')
plt.plot(df_opt['epoch'], df_opt['val_recall'], label='Limpio (OpenCV)', color='green', linewidth=2, marker='s')
plt.title('Evolución de la Sensibilidad (val_recall)')
plt.xlabel('Época')
plt.ylabel('Recall (Porcentaje de Melanomas Detectados)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 4. GRÁFICA 2: ACCURACY (Exactitud Global)
plt.subplot(1, 3, 2)
plt.plot(df_base['epoch'], df_base['val_accuracy'], label='Crudo (Baseline)', color='red', linestyle='--', marker='o')
plt.plot(df_opt['epoch'], df_opt['val_accuracy'], label='Limpio (OpenCV)', color='green', linewidth=2, marker='s')
plt.title('Evolución de la Exactitud (val_accuracy)')
plt.xlabel('Época')
plt.ylabel('Accuracy Global')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 5. GRÁFICA 3: LOSS (Función de Pérdida / Error)
plt.subplot(1, 3, 3)
plt.plot(df_base['epoch'], df_base['val_loss'], label='Crudo (Baseline)', color='red', linestyle='--', marker='o')
plt.plot(df_opt['epoch'], df_opt['val_loss'], label='Limpio (OpenCV)', color='green', linewidth=2, marker='s')
plt.title('Evolución del Error (val_loss)')
plt.xlabel('Época')
plt.ylabel('Loss (Menor es mejor)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 6. GUARDAR Y MOSTRAR RESULTADOS
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
nombre_grafico = 'comparativa_metricas.png'
plt.savefig(nombre_grafico, dpi=300)
print(f"Gráfico generado exitosamente y guardado como '{nombre_grafico}'")
plt.show()