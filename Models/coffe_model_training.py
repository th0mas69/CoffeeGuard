<<<<<<< HEAD
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# === 1️⃣ Paths ===
DATA_DIR = r"C:\Users\tluke\Desktop\CoffeGuard\Dataset"  # update if needed
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")  # optional; can be auto-split below

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# === 2️⃣ Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 80/20 split
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === 3️⃣ Print Class Indices ===
print("\n✅ Class mapping:")
print(train_gen.class_indices)
labels = list(train_gen.class_indices.keys())
print(f"\nUse this order in Flask app: LABELS = {labels}\n")

# === 4️⃣ Model Architecture ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# === 5️⃣ Compile ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 6️⃣ Train ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# === 7️⃣ Save Models ===
os.makedirs(r"C:\Users\tluke\Desktop\CoffeGuard\Models", exist_ok=True)
h5_path = r"C:\Users\tluke\Desktop\CoffeGuard\Models\best_model.h5"
tflite_path = r"C:\Users\tluke\Desktop\CoffeGuard\Models\best_model.tflite"

model.save(h5_path)
print(f"\n✅ Saved Keras model to: {h5_path}")

# === 8️⃣ Convert to TFLite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Saved TensorFlow Lite model to: {tflite_path}")
=======
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# === 1️⃣ Paths ===
DATA_DIR = r"C:\Users\tluke\Desktop\CoffeGuard\Dataset"  # update if needed
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")  # optional; can be auto-split below

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# === 2️⃣ Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 80/20 split
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === 3️⃣ Print Class Indices ===
print("\n✅ Class mapping:")
print(train_gen.class_indices)
labels = list(train_gen.class_indices.keys())
print(f"\nUse this order in Flask app: LABELS = {labels}\n")

# === 4️⃣ Model Architecture ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# === 5️⃣ Compile ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 6️⃣ Train ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# === 7️⃣ Save Models ===
os.makedirs(r"C:\Users\tluke\Desktop\CoffeGuard\Models", exist_ok=True)
h5_path = r"C:\Users\tluke\Desktop\CoffeGuard\Models\best_model.h5"
tflite_path = r"C:\Users\tluke\Desktop\CoffeGuard\Models\best_model.tflite"

model.save(h5_path)
print(f"\n✅ Saved Keras model to: {h5_path}")

# === 8️⃣ Convert to TFLite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Saved TensorFlow Lite model to: {tflite_path}")
>>>>>>> 0179464 (first commit)
