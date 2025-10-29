from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Modeli ielādē tikai pēc pirmā pieprasījuma
model = None

def get_model():
    global model
    if model is None:
        model = pipeline(
            "image-to-image",
            model="stabilityai/sd-turbo",
            safety_checker=None
        )
    return model


def whiten_teeth(image):
    """Vienkāršs, ātrs un stabils zobu balinātājs bez pilnas sejas gaišināšanas"""
    img_array = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Tiek meklēti baltie reģioni (zobi) — maska ar vieglu pelēcības pielaidi
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([179, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Mazliet palielina spilgtumu tikai maskētajos reģionos
    value = 35  # intensitāte
    hsv[..., 2] = np.clip(hsv[..., 2] + (mask > 0) * value, 0, 255)

    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(enhanced)


@app.route("/")
def home():
    return jsonify({"status": "Teeth Whitening API Light v3 😁"})


@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")

        # Balināšana (ātrā lokālā metode)
        whitened_image = whiten_teeth(image)

        # Atgriež kā JPEG
        buf = io.BytesIO()
        whitened_image.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
