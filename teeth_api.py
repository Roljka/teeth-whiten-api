from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tempfile

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh

@app.route('/')
def home():
    return jsonify({"status": "Smart Teeth Whitening API 😁 is live!"})

@app.route('/whiten', methods=['POST'])
def whiten():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe sejas punkts analīze
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return jsonify({"error": "No face detected"}), 400

            h, w, _ = img.shape
            mask = np.zeros((h, w), dtype=np.uint8)

            # paņemam mutes punktus (ap 78–88 + 308–318)
            mouth_indices = list(range(78, 89)) + list(range(308, 319))
            for face_landmarks in results.multi_face_landmarks:
                points = [(int(face_landmarks.landmark[i].x * w),
                           int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

            # nedaudz paplašinām masku
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
            mask = cv2.GaussianBlur(mask, (9, 9), 0)

            # kopējam tikai zobu reģionu
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)

            # palielinām gaišumu tikai zobu reģionā
            v_ch = np.where(mask > 0, np.clip(v_ch * 1.5, 0, 255), v_ch)
            s_ch = np.where(mask > 0, np.clip(s_ch * 0.6, 0, 255), s_ch)

            whitened_hsv = cv2.merge([h_ch, s_ch.astype(np.uint8), v_ch.astype(np.uint8)])
            whitened = cv2.cvtColor(whitened_hsv, cv2.COLOR_HSV2BGR)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, whitened)

            return send_file(temp_file.name, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
