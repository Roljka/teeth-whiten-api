from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import base64
from datetime import datetime

# ===============================
# Flask inicializācija
# ===============================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===============================
# Sākuma lapa (statuss)
# ===============================
@app.route("/")
def home():
    return "Teeth Whitening API is live! 😁"


# ===============================
# Attēla apstrāde un balināšana
# ===============================
@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Saglabā augšupielādēto attēlu
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    file.save(input_path)

    # Nolasām attēlu ar OpenCV
    image = cv2.imread(input_path)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # === Zobu balināšanas efekts (vienkāršots) ===
    # Pārvērš LAB krāsu telpā un uzlabo gaišumu
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    whitened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Neliels kontrasta un gaišuma pieaugums
    alpha = 1.15  # kontrasts
    beta = 15     # gaišums
    whitened = cv2.convertScaleAbs(whitened, alpha=alpha, beta=beta)

    # Saglabā un konvertē base64 formātā
    cv2.imwrite(output_path, whitened)
    with open(output_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # Atgriež JSON ar base64
    return jsonify({
        "message": "success",
        "image": encoded,
        "output_url": f"https://{request.host}/static/{filename}"
    })


# ===============================
# Statisko failu piegāde
# ===============================
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# ===============================
# Debug režīma palaišana (lokāli)
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
