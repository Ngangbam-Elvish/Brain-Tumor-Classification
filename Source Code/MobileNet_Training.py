#Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "../DATA/dataset/balanced_dataset"

#preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Training"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#Model
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])

    return model

model = build_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

#Save Model
os.makedirs("../Model", exist_ok=True)
model.save("../Model/mobilenet_model.h5")

#Testing
model = tf.keras.models.load_model("../Model/mobilenet_model.h5")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Testing"),
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)

print("Classification Report:\n")
print(classification_report(test_data.classes, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(test_data.classes, y_pred))
