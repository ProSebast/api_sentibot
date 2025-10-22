from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ml_model import predict_emotion
from PIL import Image
import base64
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

# Página de prueba (sólo para verificar que funciona)
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Prueba API Sentibot</title>
    </head>
    <body style="text-align:center; font-family:Arial;">
        <h2>🚀 Prueba de API de reconocimiento emocional</h2>
        <video id="video" width="300" height="200" autoplay></video>
        <p id="estado">Esperando cámara...</p>
        <canvas id="canvas" width="48" height="48" style="display:none;"></canvas>

        <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const estado = document.getElementById("estado");

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { estado.textContent = "❌ Error al activar cámara"; });

        setInterval(async () => {
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageBase64 = canvas.toDataURL("image/png");

            try {
                const res = await fetch("/predict_emotion", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: imageBase64 })
                });
                const data = await res.json();
                estado.textContent = `😀 Emoción: ${data.label} (${(data.confidence*100).toFixed(1)}%)`;
            } catch (e) {
                estado.textContent = "⚠️ Error al conectar con API";
            }
        }, 2000);
        </script>
    </body>
    </html>
    """)

# Endpoint de predicción
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
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
