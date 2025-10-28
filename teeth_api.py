from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh

@app.route("/")
def home():
    return "Teeth Whitening API – v2 😁"

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    file = request.files["file"]
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Mediapipe face mesh (precīza mute)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return jsonify({"error": "Seja netika atrasta"}), 400

        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Izvēlamies tikai iekšējās mutes zonas punktus (zobi)
        teeth_points = list(range(78, 88)) + list(range(308, 318))

        for face_landmarks in results.multi_face_landmarks:
            pts = []
            for idx in teeth_points:
                lm = face_landmarks.landmark[idx]
                pts.append((int(lm.x * w), int(lm.y * h)))

            if len(pts) > 0:
                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], 255)

        # Izgaismojam tikai zobu zonu
        result = image.copy()
        teeth_area = cv2.bitwise_and(image, image, mask=mask)
        brightened = cv2.addWeighted(teeth_area, 1.5, np.zeros_like(teeth_area), 0, 30)
        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, brightened)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp.name, result)

        return send_file(temp.name, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
