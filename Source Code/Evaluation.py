from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model('../Model/mobilenet_model.h5')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    '../DATA/dataset/balanced_dataset/Testing',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)

print(classification_report(test_data.classes, y_pred))
print(confusion_matrix(test_data.classes, y_pred))