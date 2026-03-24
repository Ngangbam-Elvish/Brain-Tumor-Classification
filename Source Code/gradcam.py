import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name="last_conv", pred_index=None):
    """
    Computes the Grad-CAM++ heatmap for a given image and model.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # First derivative (standard gradients)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Approximate 2nd and 3rd derivatives
    first_derivative = grads
    second_derivative = grads * grads
    third_derivative = grads * grads * grads

    global_sum = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))

    alpha_num = second_derivative
    alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
    # Avoid division by zero
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    
    alphas = alpha_num / alpha_denom
    
    weights = tf.maximum(first_derivative, 0.0)
    
    # Normalize alphas per map
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(1, 2))
    alpha_normalization_constant = tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, tf.ones_like(alpha_normalization_constant))
    alphas /= alpha_normalization_constant
    
    deep_linearization_weights = tf.reduce_sum((weights * alphas), axis=(1, 2))
    deep_linearization_weights = tf.reshape(deep_linearization_weights, [1, 1, -1])
    
    heatmap = tf.reduce_sum(deep_linearization_weights * last_conv_layer_output, axis=-1)
    
    # ReLU to keep only positive influence
    heatmap = tf.maximum(heatmap, 0) 
    
    # Normalize the heatmap
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0.0:
        heatmap /= max_val
        
    return heatmap.numpy()[0]

def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on the original image.
    original_image: numpy array in range 0-255, RGB format
    """
    heatmap = np.uint8(255 * heatmap)
    
    # Resize to match original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Ensure original image is uint8
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_img = np.uint8(255 * original_image)
        else:
            original_img = np.uint8(original_image)
    else:
        original_img = original_image
        
    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        
    # OpenCV uses BGR natively, so we ensure standard representation
    if original_img.shape[-1] == 3:
        # Assuming original is RGB coming from streamit, but applyColorMap gives BGR.
        # We'll treat both as BGR for the blend, then convert back to RGB.
        original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    else:
        original_img_bgr = original_img

    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, original_img_bgr, 1 - alpha, 0)
    
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
