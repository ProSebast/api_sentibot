from flask import Flask, request, jsonify, render_template_string
from ml_model import predict_emotion
import base64
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

# --- HTML de prueba ---
test_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prueba API Sentibot</title>
    <style>
        body { font-family: Arial; background: #101820; color: white; text-align: center; padding: 40px; }
        input, button { margin-top: 15px; padding: 10px; border-radius: 8px; border: none; }
        button { background-color: #007bff; color: white; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        #preview { margin-top: 20px; max-width: 250px; border-radius: 10px; }
        .card { background: #1c1f26; padding: 20px; border-radius: 15px; display: inline-block; }
    </style>
</head>
<body>
    <h1>🚀 API Sentibot - Test</h1>
    <div class="card">
        <p>Sube una imagen con una expresión facial:</p>
        <input type="file" id="fileInput" accept="image/*"><br>
        <img id="preview" src="" alt="">
        <br><button onclick="sendImage()">Enviar a API</button>
        <p id="result"></p>
    </div>

    <script>
        const apiUrl = "/predict_emotion";
        const fileInput = document.getElementById("fileInput");
        const preview = document.getElementById("preview");
        const result = document.getElementById("result");

        fileInput.addEventListener("change", () => {
            const file = fileInput.files[0];
            if (file) preview.src = URL.createObjectURL(file);
        });

        async function sendImage() {
            const file = fileInput.files[0];
            if (!file) {
                result.textContent = "Por favor, selecciona una imagen primero.";
                return;
            }
            const reader = new FileReader();
            reader.onload = async function () {
                const base64 = reader.result;
                result.textContent = "Enviando imagen...";
                const response = await fetch(apiUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: base64 })
                });
                const data = await response.json();
                result.innerHTML = `<b>Emoción:</b> ${data.label}<br><b>Confianza:</b> ${data.confidence}`;
            };
            reader.readAsDataURL(file);
        }
    </script>
</body>
</html>
"""

# --- RUTA PRINCIPAL: muestra el HTML ---
@app.route("/")
def index():
    return render_template_string(test_html)

# --- ENDPOINT del modelo ---
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
✅ Cómo usarlo: