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

@app.route('/')
def home():
    return jsonify({"message": "Teeth Whitening API v3 — only teeth 😁"})


@app.route('/whiten', methods=['POST'])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        image = cv2.imread(tmp.name)

    if image is None:
        return jsonify({"error": "Nevar nolasīt attēlu"}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    # Mediapipe sejas tīkls
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6
    ) as face_mesh:

        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"error": "Seja netika atrasta"}), 400

        # Izmantojam tikai mutes zonas punktus
        mouth_indices = list(range(61, 91))  # Mutes reģions
        for face_landmarks in results.multi_face_landmarks:
            mouth_points = []
            for idx in mouth_indices:
                x = int(face_landmarks.landmark[idx].x * width)
                y = int(face_landmarks.landmark[idx].y * height)
                mouth_points.append((x, y))

            # Izveido masku zobu zonai
            mouth_points = np.array(mouth_points, dtype=np.int32)
            cv2.fillConvexPoly(mask, mouth_points, 255)

    # Nedaudz paplašina un izpludina masku (dabiskāk)
    mask = cv2.GaussianBlur(mask, (15, 15), 10)

    # Izveido gaišāku versiju tikai zobu zonai
    teeth_area = cv2.bitwise_and(image, image, mask=mask)
    whitened_teeth = cv2.convertScaleAbs(teeth_area, alpha=1.1, beta=35)

    # Apvieno ar oriģinālu
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(image, image, mask=inv_mask)
    result = cv2.add(background, whitened_teeth)

    output_path = os.path.join(tempfile.gettempdir(), "whitened.jpg")
    cv2.imwrite(output_path, result)
    return send_file(output_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
