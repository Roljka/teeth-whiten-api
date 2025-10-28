from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)  # ļauj frontendam no jebkuras lapas sūtīt pieprasījumus

# Maksimālais faila izmērs – 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

@app.route("/")
def home():
    return jsonify({"status": "🦷 Teeth Whitening API is live!"})

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nav atrasts fails"}), 400

        file = request.files["file"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            image_path = tmp.name

        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Nederīgs attēla formāts"}), 400

        # Samazina lielas bildes
        if image.shape[1] > 800:
            ratio = 800 / image.shape[1]
            new_size = (800, int(image.shape[0] * ratio))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Pielieto vienkāršu gaišināšanas efektu
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        whitened = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        result_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(result_path, whitened)

        return send_file(result_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if "image_path" in locals() and os.path.exists(image_path):
                os.remove(image_path)
            if "result_path" in locals() and os.path.exists(result_path):
                os.remove(result_path)
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
