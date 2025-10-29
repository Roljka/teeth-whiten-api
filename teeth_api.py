from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tempfile, os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Teeth Whitening API v7 🦷"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    file = request.files["file"]
    intensity = int(request.form.get("intensity", 25))  # 10–50

    # Saglabā pagaidu failu
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        img = cv2.imread(tmp.name)

    if img is None:
        return jsonify({"error": "Nevar nolasīt attēlu"}), 400

    # Konvertē uz HSV krāstelpu
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Zobi = gaiši pikseļi ar zemu piesātinājumu (S)
    lower = np.array([0, 0, 160])     # apakšējais slieksnis (tumšākais tonis)
    upper = np.array([180, 60, 255])  # augšējais slieksnis (baltie reģioni)
    mask = cv2.inRange(hsv, lower, upper)

    # Izpludina malu, lai maska būtu gluda
    mask = cv2.GaussianBlur(mask, (15, 15), 5)

    # Balina tikai ar masku nosegtās vietas
    img_whitened = img.copy()
    img_whitened[mask > 0] = cv2.add(img[mask > 0], (intensity, intensity, intensity))

    out_path = os.path.join(tempfile.gettempdir(), "whitened_v7.jpg")
    cv2.imwrite(out_path, img_whitened)
    return send_file(out_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
