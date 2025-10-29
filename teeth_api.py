from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

def whiten_teeth_only(image_pil):
    # pārvērš PIL -> numpy RGB
    img_rgb = np.array(image_pil.convert("RGB"))
    h, w, _ = img_rgb.shape

    # Inicializē FaceMesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            print("❌ Seju neatrod.")
            return image_pil  # Atgriež sākotnējo, ja neko neatpazīst

        # Izveido masku
        mask = np.zeros((h, w), dtype=np.uint8)

        # Piemēro muti (ap lūpām + zobi)
        for face_landmarks in results.multi_face_landmarks:
            mouth_points = [
                61, 76, 78, 80, 82, 84, 13, 312, 310, 308, 402, 324, 318, 14
            ]
            pts = np.array([
                (int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h))
                for i in mouth_points
            ], np.int32)

            # aizpilda muti ar baltu zonu (mask)
            cv2.fillPoly(mask, [pts], 255)

        # Izbalina tikai maskā
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_, s_, v_ = cv2.split(hsv)

        # Pielieto tikai maskētajā zonā
        whitened_v = v_.astype(np.float32)
        whitened_v[mask == 255] = np.clip(whitened_v[mask == 255] * 1.45, 0, 255)

        v_ = whitened_v.astype(np.uint8)
        hsv_whitened = cv2.merge([h_, s_, v_])
        img_result = cv2.cvtColor(hsv_whitened, cv2.COLOR_HSV2RGB)

        return Image.fromarray(img_result)


@app.route('/whiten', methods=['POST'])
def whiten_image():
    if 'file' not in request.files:
        return jsonify({"error": "Nav augšupielādēts fails"}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    result_image = whiten_teeth_only(image)

    # saglabā rezultātu kā JPG atmiņā
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG', quality=95)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/')
def home():
    return jsonify({"status": "🦷 Teeth Whitening API darbojas!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
