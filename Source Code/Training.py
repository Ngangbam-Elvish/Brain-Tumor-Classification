#Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "../DATA/dataset/balanced_dataset"

#preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
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
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(128, (3,3), activation='relu', name='last_conv'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4, activation='softmax'))

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
model.save("../Model/cnn_model.h5")

#Testing
model = tf.keras.models.load_model("../Model/cnn_model.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

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