from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh

@app.route("/")
def home():
    return "Teeth Whitening API – precision mode 🦷✨"

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    file = request.files["file"]
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Sejas noteikšana
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return jsonify({"error": "Seja netika atrasta"}), 400

        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Mutes iekšējā daļa (kur zobi)
        mouth_points = list(range(78, 88)) + list(range(308, 318))

        for face_landmarks in results.multi_face_landmarks:
            pts = np.array(
                [(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark) if i in mouth_points],
                np.int32
            )
            cv2.fillPoly(mask, [pts], 255)

        # Izgaismojam tikai balto zonu mutē
        mouth_area = cv2.bitwise_and(image, image, mask=mask)
        hsv = cv2.cvtColor(mouth_area, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        teeth_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Maigi pastiprinām gaišumu zobiem
        teeth_area = cv2.bitwise_and(image, image, mask=teeth_mask)
        brighter_teeth = cv2.addWeighted(teeth_area, 1.4, np.zeros_like(teeth_area), 0, 25)

        # Salīmējam kopā tikai zobiņu zonu
        inv_mask = cv2.bitwise_not(teeth_mask)
        result = cv2.bitwise_and(image, image, mask=inv_mask)
        result = cv2.add(result, brighter_teeth)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp.name, result)

        return send_file(temp.name, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
