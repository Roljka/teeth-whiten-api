from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tempfile, os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Teeth Whitening API v6 😁"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    file = request.files["file"]
    intensity = int(request.form.get("intensity", 25))  # 10–50

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        img = cv2.imread(tmp.name)

    if img is None:
        return jsonify({"error": "Nevar nolasīt attēlu"}), 400

    # pārvērš LAB krāstelpā (L = gaišums)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # izveido masku gaišajiem, dzeltenīgiem reģioniem (zobi)
    mask = cv2.inRange(lab, (150, 120, 120), (255, 145, 160))
    mask = cv2.GaussianBlur(mask, (25, 25), 10)

    # palielina gaišumu tikai zobiem
    l = cv2.add(l, intensity, mask=mask)
    lab = cv2.merge((l, a, b))
    whitened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    out_path = os.path.join(tempfile.gettempdir(), "whitened_v6.jpg")
    cv2.imwrite(out_path, whitened)
    return send_file(out_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
