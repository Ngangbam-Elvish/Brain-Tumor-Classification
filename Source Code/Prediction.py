import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from gradcam import make_gradcam_plus_plus_heatmap, overlay_heatmap

def build_model():
    inputs = layers.Input(shape=(224,224,3))
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu', name='last_conv')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

model = build_model()
model.load_weights('../Model/cnn_model.h5')

# The classes are alphanumerically sorted by ImageDataGenerator
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    original_img = cv2.resize(img, (224,224))
    img_array = original_img / 255.0
    img_array = np.reshape(img_array, (1,224,224,3))

    pred = model.predict(img_array)
    pred_idx = np.argmax(pred[0])
    label = classes[pred_idx]
    
    heatmap = make_gradcam_plus_plus_heatmap(img_array, model, "last_conv", pred_idx)
    overlay = overlay_heatmap(heatmap, original_img)
        
    return label, overlay

if __name__ == "__main__":
    # Test the prediction if you have a sample.jpg
    import os
    if os.path.exists('sample.jpg'):
        label, overlay = predict_image('sample.jpg')
        print(f"Predicted: {label}.")
        if overlay is not None:
            cv2.imwrite('sample_gradcam.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print("Grad-CAM saved to sample_gradcam.jpg")