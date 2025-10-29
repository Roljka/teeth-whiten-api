from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

def whiten_teeth(image):
    """Atrod zobus un balina tikai tos"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=25, minSize=(40, 40))

    mask = np.zeros_like(gray)
    for (x, y, w, h) in smiles:
        cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, int(h / 2.5)), 0, 0, 360, 255, -1)

    result = image.copy()
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, (mask > 0).astype(np.uint8) * 40)
    final_hsv = cv2.merge((h, s, np.clip(v, 0, 255)))
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 11)
    result = np.where(blurred_mask[..., None] > 0, result, image)
    return result

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "Teeth Whitening API v3 🦷 is live!"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēts fails"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    processed_image = whiten_teeth(image)
    _, buffer = cv2.imencode(".jpg", processed_image)
    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
