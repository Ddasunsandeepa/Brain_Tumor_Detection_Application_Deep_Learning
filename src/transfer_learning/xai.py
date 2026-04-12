import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "../../model/brain_tumor_model_tf.keras"

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def explain_image(img_path):
    model = load_model(MODEL_PATH)

    # Fix layer index
    base_model = model.layers[0]
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
    heatmap /= np.max(heatmap)

    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + img_cv

    plt.imshow(overlay.astype('uint8'))
    plt.title("Model Attention")
    plt.axis('off')
    plt.show()

# Example
if __name__ == "__main__":
    explain_image("../../data/Testing/glioma/Te-gl_1.jpg")