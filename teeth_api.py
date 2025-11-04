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

# ---------- MediaPipe ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# Iekšējās lūpas landmarķi
INNER_LIP_IDX = np.array([78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95], dtype=np.int32)

# ---------- Noklusētie TUNING (ja query nav) ----------
DEF_DIL_H_SCALE = 0.060   # horizontāli
DEF_DIL_V_SCALE = 0.025   # vertikāli
DEF_EDGE_GUARD   = 4      # px
DEF_FEATHER_PX   = 15
DEF_A_MAX        = 148    # LAB A (rozā/sarkans)
DEF_RED_H_LOW    = 12     # HSV red
DEF_RED_H_HIGH   = 170
DEF_RED_S_MIN    = 28
DEF_L_DELTA      = -10    # L thr korekcija
DEF_B_DELTA      = +18    # B thr korekcija
DEF_MIN_TOOTH_CC = 80

def _getf(name, default):
    v = request.args.get(name, None)
    if v is None: return float(default)
    try: return float(v)
    except: return float(default)

def _geti(name, default):
    v = request.args.get(name, None)
    if v is None: return int(default)
    try: return int(v)
    except: return int(default)

def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask, k=11):
    k = int(k)
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(mask, (k, k), 0)

def _build_mouth_mask(img_bgr, landmarks):
    # parametri no query (vai default)
    DIL_H_SCALE = _getf("dilH", DEF_DIL_H_SCALE)
    DIL_V_SCALE = _getf("dilV", DEF_DIL_V_SCALE)
    EDGE_GUARD  = _geti("edge", DEF_EDGE_GUARD)
    FEATHER_PX  = _geti("feather", DEF_FEATHER_PX)

    h, w = img_bgr.shape[:2]
    inner = _landmarks_to_xy(landmarks, w, h, INNER_LIP_IDX)

    area = cv2.contourArea(inner)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    side = max(1.0, math.sqrt(area))
    kx = max(3, int(round(side * DIL_H_SCALE))) | 1
    ky = max(3, int(round(side * DIL_V_SCALE))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    mask = cv2.dilate(mask, kernel, iterations=1)

    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_safe = cv2.erode(mask, guard_k, iterations=1)

    return _smooth_mask(inner_safe, FEATHER_PX)

def _build_teeth_mask(img_bgr, mouth_mask):
    # parametri no query
    A_MAX        = _geti("amax", DEF_A_MAX)
    RED_H_LOW    = _geti("redLow", DEF_RED_H_LOW)
    RED_H_HIGH   = _geti("redHigh", DEF_RED_H_HIGH)
    RED_S_MIN    = _geti("redS", DEF_RED_S_MIN)
    L_DELTA      = _geti("ld", DEF_L_DELTA)
    B_DELTA      = _geti("bd", DEF_B_DELTA)
    EDGE_GUARD   = _geti("edge", DEF_EDGE_GUARD)

    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, _V = cv2.split(hsv)

    m = mouth_mask > 0
    if not np.any(m):
        return np.zeros_like(mouth_mask)

    L_thr, _ = cv2.threshold(L[m].astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    B_thr, _ = cv2.threshold(B[m].astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    L_thr = max(40, int(L_thr) + int(L_DELTA))
    B_thr = min(240, int(B_thr) + int(B_DELTA))

    red_hsv = (((H <= RED_H_LOW) | (H >= RED_H_HIGH)) & (S >= RED_S_MIN))
    red_lab = (A >= A_MAX)
    red_like = (red_hsv | red_lab)

    raw = ((L > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_safe = cv2.erode(mouth_mask, guard_k, iterations=1)
    raw = cv2.bitwise_and(raw, inner_safe)

    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)

    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    MIN_TOOTH_CC = _geti("mincc", DEF_MIN_TOOTH_CC)
    for c in cnts:
        if cv2.contourArea(c) > MIN_TOOTH_CC:
            cv2.drawContours(filled, [c], -1, 255, -1)

    return _smooth_mask(filled, _geti("feather", DEF_FEATHER_PX))

def _teeth_whiten(img_bgr):
    FEATHER_PX = _geti("feather", DEF_FEATHER_PX)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    teeth_mask = _build_teeth_mask(img_bgr, mouth_mask)

    if np.sum(teeth_mask) == 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0

    Lf = L.astype(np.float32); Bf = B.astype(np.float32)
    Lf[m] = np.clip(Lf[m] * 1.15 + 12, 0, 255)
    Bf[m] = np.clip(Bf[m] * 0.82 - 8, 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    out_bgr[m] = blur[m]
    return out_bgr

def _read_image_from_request():
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
