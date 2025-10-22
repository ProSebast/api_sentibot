from flask import Flask, request, jsonify
from ml_model import predict_emotion
import base64
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "API de reconocimiento emocional activa ✅"})

@app.route("/predict_emotion", methods=["POST"])
def predict():
    data = request.get_json()
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"label": "Null", "confidence": 0})

    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("L")
        label, confidence = predict_emotion(image)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"label": "Error", "confidence": 0, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
