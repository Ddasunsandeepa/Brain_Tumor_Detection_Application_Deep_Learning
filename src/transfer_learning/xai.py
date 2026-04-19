# src/transfer_learning/xai.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../model/brain_tumor_model_tf.keras")
)
model = load_model(MODEL_PATH)

def generate_heatmap(img_path, output_path):
    global model

    # Find EfficientNet inside model
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNet base model not found!")

    # Get last conv layer
    last_conv_layer = base_model.get_layer("top_conv")

    activation_model = Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    img = load_img(img_path, target_size=(224,224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    feature_maps = activation_model.predict(img_array)

    heatmap = np.mean(feature_maps[0], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + img_cv

    cv2.imwrite(output_path, overlay)

    return output_path