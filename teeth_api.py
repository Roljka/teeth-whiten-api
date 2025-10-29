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
    return jsonify({"message": "Teeth Whitening API v5 😁"})


@app.route('/whiten', methods=['POST'])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    intensity = int(request.form.get("intensity", 30))  # default 30
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
        min_detection_confidence=0.6
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"error": "Seja netika atrasta"}), 400

        # tikai zobu punkti (mediapipe indeksi — precīzi ap zobiem)
        teeth_idx = [78, 80, 82, 13, 312, 310, 308, 324, 318, 87, 84, 14]
        for face_landmarks in results.multi_face_landmarks:
            teeth_points = []
            for idx in teeth_idx:
                x = int(face_landmarks.landmark[idx].x * width)
                y = int(face_landmarks.landmark[idx].y * height)
                teeth_points.append((x, y))

            if len(teeth_points) > 0:
                cv2.fillPoly(mask, [np.array(teeth_points, dtype=np.int32)], 255)

    # izpludina malu, lai nav “robains” efekts
    mask = cv2.GaussianBlur(mask, (25, 25), 20)

    # izgūst tikai zobu daļu
    teeth = cv2.bitwise_and(original, original, mask=mask)

    # balināšana LAB krāsu telpā
    lab = cv2.cvtColor(teeth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, intensity)  # regulējams balinājums
    lab = cv2.merge((l, a, b))
    whitened_teeth = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # sapludina ar fonu
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(original, original, mask=inv_mask)
    final_image = cv2.add(background, cv2.bitwise_and(whitened_teeth, whitened_teeth, mask=mask))

    output_path = os.path.join(tempfile.gettempdir(), "whitened_v5.jpg")
    cv2.imwrite(output_path, final_image)
    return send_file(output_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
