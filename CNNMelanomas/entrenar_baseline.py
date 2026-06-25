import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.utils.class_weight import compute_class_weight

# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# 2. GENERADORES DE DATOS (Ingesta)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

print("Cargando datos de entrenamiento...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' 
)

print("Cargando datos de validación...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# 3. CÁLCULO DE PESOS DE CLASE
etiquetas_entrenamiento = train_generator.classes

pesos = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(etiquetas_entrenamiento),
    y=etiquetas_entrenamiento
)
pesos_clase = {0: pesos[0], 1: pesos[1]}

print(f"Pesos calculados -> Benigno: {pesos_clase[0]:.2f}, Melanoma: {pesos_clase[1]:.2f}")

# 4. CONSTRUCCIÓN DE LA ARQUITECTURA (Transfer Learning)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelamos las capas base para no destruir el conocimiento previo
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) 
prediccion = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=prediccion)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

# 5. ENTRENAMIENTO Y CALLBACKS
checkpoint = ModelCheckpoint(
    'baseline_resnet50.h5', 
    monitor='val_recall', 
    save_best_only=True, 
    mode='max',
    verbose=1
)

# Detener si deja de mejorar después de 5 épocas
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

# Guardar las métricas en un archivo CSV
csv_logger = CSVLogger(
    'metricas_baseline.csv', 
    append=False, 
    separator=','
)

print("\nIniciando el entrenamiento del Modelo Base...")
historial = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=pesos_clase,
    callbacks=[checkpoint, early_stopping,csv_logger]
)

print("\nEntrenamiento finalizado. Modelo 'baseline_resnet50.h5' guardado.")