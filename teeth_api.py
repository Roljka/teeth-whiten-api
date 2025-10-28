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
    return jsonify({"message": "Teeth Whitening API v3.1 — clean face edition 😁"})


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

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"error": "Seja netika atrasta"}), 400

        mouth_indices = list(range(61, 91))  # Tikai mutes apgabals
        for face_landmarks in results.multi_face_landmarks:
            mouth_points = []
            for idx in mouth_indices:
                x = int(face_landmarks.landmark[idx].x * width)
                y = int(face_landmarks.landmark[idx].y * height)
                mouth_points.append((x, y))

            if len(mouth_points) > 0:
                mouth_points = np.array(mouth_points, dtype=np.int32)
                cv2.fillConvexPoly(mask, mouth_points, 255)

    # Izpludina robežas — dabiskāk
    mask = cv2.GaussianBlur(mask, (25, 25), 20)

    # Izgūst tikai zobu laukumu
    teeth_area = cv2.bitwise_and(image, image, mask=mask)

    # Konvertē uz LAB krāsu telpu un paceļ L kanālu (gaišumu)
    lab = cv2.cvtColor(teeth_area, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 25)  # balināšanas intensitāte
    lab = cv2.merge((l, a, b))
    whitened_teeth = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Saglabā tikai zobu laukumu un pārējo atstāj neskartu
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(image, image, mask=inv_mask)
    result = cv2.add(background, cv2.bitwise_and(whitened_teeth, whitened_teeth, mask=mask))

    output_path = os.path.join(tempfile.gettempdir(), "whitened.jpg")
    cv2.imwrite(output_path, result)
    return send_file(output_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
