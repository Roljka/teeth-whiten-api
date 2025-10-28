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
    return jsonify({"message": "Teeth Whitening API v4.2 — true teeth detection 😁"})


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

    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"error": "Seja netika atrasta"}), 400

        # Zobu reģions — augšējie un apakšējie zobi (vidusdaļa)
        upper_teeth_idx = [78, 79, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lower_teeth_idx = [88, 87, 86, 85, 84, 14, 317, 318, 319, 403, 324]
        teeth_indices = upper_teeth_idx + lower_teeth_idx

        for face_landmarks in results.multi_face_landmarks:
            teeth_points = []
            for idx in teeth_indices:
                x = int(face_landmarks.landmark[idx].x * width)
                y = int(face_landmarks.landmark[idx].y * height)
                teeth_points.append((x, y))

            if len(teeth_points) > 0:
                cv2.fillPoly(mask, [np.array(teeth_points, dtype=np.int32)], 255)

    # Izpludina zobu maskas robežas
    mask = cv2.GaussianBlur(mask, (15, 15), 10)

    # Apstrādā tikai zobu zonu
    teeth = cv2.bitwise_and(original, original, mask=mask)

    lab = cv2.cvtColor(teeth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 35)  # intensitāte — 20 dabīgi, 35 perfekti
    lab = cv2.merge((l, a, b))
    whitened_teeth = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(original, original, mask=inv_mask)
    final_image = cv2.add(background, cv2.bitwise_and(whitened_teeth, whitened_teeth, mask=mask))

    output_path = os.path.join(tempfile.gettempdir(), "whitened_teeth.jpg")
    cv2.imwrite(output_path, final_image)
    return send_file(output_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
