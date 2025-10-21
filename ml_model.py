from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from PIL import Image

# Carga del modelo solo una vez
model = load_model("model/sentibotv2.h5")
EMOTIONS = ["Feliz", "Triste", "Neutral", "Enojado", "Sorprendido"]

def predict_emotion(image: Image.Image):
    image = image.resize((48,48))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if img_array.shape[-1] != 1:
        img_array = img_array.mean(axis=-1, keepdims=True)  # escalar a gris si no lo es

    preds = model.predict(img_array)
    label_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return EMOTIONS[label_idx], confidence
