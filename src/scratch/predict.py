import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "../../model/scratch_model.keras"

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

model = load_model(MODEL_PATH)

def predict_image(img_path):
    # model = load_model(MODEL_PATH)

    img = load_img(img_path, target_size=(224,224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    label = class_names[predicted_class]

    if label == "notumor":
        result = "No Tumor"
    else:
        result = f"Tumor: {label}"

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence*100:.2f}%")

    plt.imshow(img)
    plt.title(f"{result} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.savefig("prediction.png")

# Run
if __name__ == "__main__":
    predict_image("../../data/Testing/glioma/Te-gl_1.jpg")