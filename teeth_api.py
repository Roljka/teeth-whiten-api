import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Mediapipe init – statisks režīms, 1 seja, zemāks resurss
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,   # pietiek zobi/mutei
    min_detection_confidence=0.5
)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Pillow -> OpenCV BGR"""
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600) -> np.ndarray:
    """Nolasa bildi, labo EXIF rotāciju, samazina max izmēru taupot RAM/CPU."""
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)  # salabo EXIF orientāciju
    # Samazinām lielus attēlus
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_to_bgr(img)

def lips_mask_from_landmarks(h, w, landmarks) -> np.ndarray:
    """
    Uztaisa mutes/lūpu masku no FaceMesh 468 punktiem.
    Varam izmantot FACEMESH_LIPS savienojumus -> unikālie punkti -> convex hull.
    """
    lips_connections = mp_face_mesh.FACEMESH_LIPS
    idx = set()
    for a, b in lips_connections:
        idx.add(a)
        idx.add(b)
    pts = []
    for i in idx:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask

def build_teeth_mask(bgr: np.ndarray, lips_mask: np.ndarray) -> np.ndarray:
    """
    Precizē zobu masku tikai mutē:
    - HSV filtri: zobi parasti ir augstāks V (gaišāki) + zemāks S (mazāk krāsas)
    - morfoloģija, lai atdalītu no smaganām/lūpām.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Pamatmaskas iekš mutes
    mouth = lips_mask > 0

    # Pragmatiskas robežas (strādā iekštelpās ar siltu gaismu):
    # zems piesātinājums (s < 90..110) un pietiekama gaišuma (v > 130..150)
    teeth_candidate = (s < 100) & (v > 135) & mouth

    # Tīrīšana – aizvācam trokšņus, saaugam zobu laukumus
    kernel = np.ones((3, 3), np.uint8)
    mask = np.zeros_like(lips_mask)
    mask[teeth_candidate] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Nedaudz “iekšpusē”, lai neskartu smaganu/lūpu robežas
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask

def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 16, b_shift: int = 20) -> np.ndarray:
    """
    Balināšana LAB telpā:
    - Palielinām L (gaišums)
    - Samazinām b* (dzeltenumu) => b = b - b_shift
    Darbojamies tikai zobu maskā (3 kanāli).
    """
    if np.count_nonzero(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # tikai maskā:
    mask = teeth_mask > 0
    L_new = L.astype(np.int16)
    B_new = B.astype(np.int16)

    L_new[mask] = np.clip(L_new[mask] + l_gain, 0, 255)
    B_new[mask] = np.clip(B_new[mask] - b_shift, 0, 255)

    lab2 = cv2.merge([L_new.astype(np.uint8), A, B_new.astype(np.uint8)])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data with field 'file'."), 400

        bgr = load_image_fix_orientation(request.files["file"])

        h, w = bgr.shape[:2]
        # FaceMesh
        results = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = results.multi_face_landmarks[0].landmark
        lips_mask = lips_mask_from_landmarks(h, w, landmarks)

        # Teeth mask only inside mouth
        teeth_mask = build_teeth_mask(bgr, lips_mask)

        out = whiten_only_teeth(bgr, teeth_mask, l_gain=16, b_shift=24)

        # Uzrakstām JPEG atmiņā
        _, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="whitened.jpg"
        )
    except Exception as e:
        # sniedzam sakarīgu kļūdu
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
