import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from gradcam import make_gradcam_plus_plus_heatmap, overlay_heatmap

# The classes are alphanumerically sorted by ImageDataGenerator
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Model Loading Definitions ---
@st.cache_resource
def load_cnn_model():
    # functional api approach to fix gradcam output graphing
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
    model = models.Model(inputs=inputs, outputs=outputs)
    model.load_weights('../Model/cnn_model.h5')
    return model

@st.cache_resource
def load_mobilenet_model():
    model = tf.keras.models.load_model('../Model/mobilenet_model.h5')
    return model

# Streamlit UI App
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("Brain Tumor Detection")
st.write("Upload an MRI image to get a prediction and visualize tumor regions using deep learning.")

st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a Model Strategy:",
    ("Custom CNN (with Grad-CAM++)", "MobileNetV2 (High Accuracy)")
)

file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    st.write("Processing image...")
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    original_img = cv2.resize(img, (224,224))
    
    if model_choice == "Custom CNN (with Grad-CAM++)":
        # Load custom CNN
        model = load_cnn_model()
        
        # Preprocessing for Custom CNN
        img_array = original_img / 255.0
        img_array = np.reshape(img_array, (1, 224, 224, 3))
        
        # Prediction
        pred = model.predict(img_array)
        pred_idx = np.argmax(pred[0])
        label = classes[pred_idx]
        confidence = float(pred[0][pred_idx])
        
        # Grad-CAM++
        heatmap = make_gradcam_plus_plus_heatmap(img_array, model, "last_conv", pred_idx)
        overlay = overlay_heatmap(heatmap, original_img)
        
        st.success(f"**CNN Prediction**: {label.upper()} (Confidence: {confidence:.2%})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM++ Visual Explanation", use_container_width=True)

    else:
        # Load MobileNetV2
        model = load_mobilenet_model()
        
        # Preprocessing for MobileNetV2
        img_array = np.float32(original_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Prediction
        pred = model.predict(img_array)
        pred_idx = np.argmax(pred[0])
        label = classes[pred_idx]
        confidence = float(pred[0][pred_idx])
        
        st.success(f"**MobileNetV2 Prediction**: {label.upper()} (Confidence: {confidence:.2%})")
        
        # Display image without Grad-CAM
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=False)
        st.info("Grad-CAM++ is natively incompatible with nested models. Falling back to simple prediction mode.")