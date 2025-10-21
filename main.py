from flask import Flask, request, jsonify
from ml_model import predict_emotion
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/predict_emotion", methods=["POST"])
def predict():
    data = request.get_json()
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"label": "Null", "confidence": 0})

    image_bytes = base64.b64decode(image_base64.split(",")[1])
    image = Image.open(BytesIO(image_bytes)).convert("L")
    label, confidence = predict_emotion(image)
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
