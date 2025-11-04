import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ---------- MediaPipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# Iekšējās lūpas landmarķi (468 shēmā)
INNER_LIP_IDX = np.array([
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,
    87,178,88,95
], dtype=np.int32)

# ---------- Palīgfunkcijas ----------
def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask, k=11):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask, (k, k), 0)

def _build_mouth_mask(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]
    inner = _landmarks_to_xy(landmarks, w, h, INNER_LIP_IDX)

    area = cv2.contourArea(inner)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    # Mazāka paplašināšana; NEbīdam uz leju (mazāk lūpu paķeršanas)
    dil = max(8, int(math.sqrt(area) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask = _smooth_mask(mask, 17)
    return mask

def _build_teeth_mask(img_bgr, mouth_mask):
    """Cieša zobu maska mutes iekšienē (bez lūpām/smaganām)."""
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    m = mouth_mask > 0
    Lm = L[m]; Bm = B[m]

    if Lm.size == 0:
        return np.zeros_like(mouth_mask)

    # Relatīvi sliekšņi pret mutes zonu
    L_thr = np.percentile(Lm, 60)     # zobi – gaišāki
    B_thr = np.percentile(Bm, 55)     # mazāk dzelta
    A_max = 150                       # izslēdz rozā (smaganas)

    teeth0 = ((L > L_thr) & (B < B_thr) & (A < A_max) & m).astype(np.uint8) * 255

    # Edge-guard: novācam mutes iekšējās malas joslu
    EDGE_GUARD = 4
    k_guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_shrunk = cv2.erode(mouth_mask, k_guard, iterations=1)
    teeth1 = cv2.bitwise_and(teeth0, inner_shrunk)

    # Trokšņu tīrīšana + viegla pievilkšana pie zoba kontūras
    teeth1 = cv2.morphologyEx(teeth1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    ERODE_PX = 2
    k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX*2+1, ERODE_PX*2+1))
    teeth1 = cv2.erode(teeth1, k_er, iterations=1)
    teeth1 = cv2.morphologyEx(teeth1, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    return _smooth_mask(teeth1, 15)

def _teeth_whiten(img_bgr):
    # MediaPipe
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark

    # 1) mutes maska
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    # 2) zobu maska
    teeth_mask = _build_teeth_mask(img_bgr, mouth_mask)
    if np.sum(teeth_mask) == 0:
        return img_bgr

    # 3) balināšana LAB telpā (tikai maskā)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    Lf = L.astype(np.float32)
    Bf = B.astype(np.float32)
    m = teeth_mask > 0

    Lf[m] = np.clip(Lf[m] * 1.15 + 12, 0, 255)   # gaišāk
    Bf[m] = np.clip(Bf[m] * 0.82 - 8, 0, 255)    # mazāk dzeltena

    L2 = Lf.astype(np.uint8)
    B2 = Bf.astype(np.uint8)
    lab_out = cv2.merge([L2, A, B2])

    out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # Neliels izlīdzinājums tikai maskā (plankumu novākšanai)
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    out_bgr[m] = blur[m]

    return out_bgr

def _read_image_from_request():
    # pieņem 'file' VAI 'image'
    if 'file' in request.files:
        f = request.files['file']
    elif 'image' in request.files:
        f = request.files['image']
    else:
        return None, ("missing file field 'file' (multipart/form-data)", 400)

    try:
        pil = Image.open(f.stream)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        return None, (f"cannot open image: {e}", 400)

    img = np.array(pil)[:, :, ::-1]  # RGB->BGR
    return img, None

# ---------- API ----------
@app.route("/whiten", methods=["POST"])
def whiten():
    img_bgr, err = _read_image_from_request()
    if err:
        return jsonify({"error": err[0]}), err[1]

    out_bgr = _teeth_whiten(img_bgr)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(out_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
