# requirements: flask opencv-python mediapipe numpy pillow
# pip install flask opencv-python mediapipe numpy pillow

from flask import Flask, request, send_file
import cv2, numpy as np
from io import BytesIO
import mediapipe as mp

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh

def whiten_teeth_region(img_bgr, landmarks, intensity=1.4):
    h, w = img_bgr.shape[:2]
    # Zobu apgabals: mediapipe sejas mesh indeksu intervāls ap muti
    mouth_idxs = list(range(78, 88)) + list(range(308, 318))
    pts = []
    for i in mouth_idxs:
        lm = landmarks.landmark[i]
        pts.append((int(lm.x*w), int(lm.y*h)))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((15,15),np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (25,25), 0)

    # pārvērš uz LAB un gaišina tikai maskā
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)
    L2 = np.clip(L * (1 + (mask.astype(np.float32)/255)*(intensity-1)), 0, 255).astype(np.uint8)
    img_lab = cv2.merge([L2, A, B])
    result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return result

@app.route('/teeth-whiten', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No file", 400

    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "Invalid image", 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            # ja nav atrasta seja, atdod oriģinālo
            _, buf = cv2.imencode(".jpg", img)
            return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")

        face_landmarks = results.multi_face_landmarks[0]
        whitened = whiten_teeth_region(img, face_landmarks)

        _, buf = cv2.imencode(".jpg", whitened, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
