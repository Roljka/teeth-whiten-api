import io
import math
import os
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import mediapipe as mp

# -----------------------
# App
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# MediaPipe FaceMesh
# -----------------------
mp_face = mp.solutions.face_mesh
FACE_MESH = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # precīzāki lūpu punkti
    min_detection_confidence=0.5
)
# izmantojam gatavo lūpu malu savienojumu kopu un iegūstam indeksus
LIP_CONN = mp_face.FACEMESH_LIPS
LIP_IDXS = sorted({i for edge in LIP_CONN for i in edge})

# -----------------------
# Palīgfunkcijas
# -----------------------
def _landmarks_to_xy(landmarks, w, h, idxs) -> np.ndarray:
    pts = []
    for i in idxs:
        p = landmarks[i]
        pts.append((int(p.x * w), int(p.y * h)))
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask: np.ndarray, k: int = 21) -> np.ndarray:
    if k % 2 == 0:
        k += 1
    m = cv2.GaussianBlur(mask, (k, k), 0)
    return m

def _build_mouth_mask(img_bgr: np.ndarray, landmarks, pad_k: float = 0.026, shift_k: float = 1/3) -> np.ndarray:
    """
    Veido mutes (zobu) masku no LŪPU punktiem:
    - aizpilda convex hull
    - neliels dilate + neliels erode, lai robeža pieguļ (mazāks "padding")
    - neliela vertikāla nobīde uz leju (prom no augšlūpas)
    """
    h, w = img_bgr.shape[:2]
    lip_pts = _landmarks_to_xy(landmarks, w, h, LIP_IDXS)

    if len(lip_pts) < 6:
        return np.zeros((h, w), dtype=np.uint8)

    hull = cv2.convexHull(lip_pts)
    base = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(base, hull, 255)

    area = cv2.contourArea(hull)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    dil = max(6, int(math.sqrt(area) * pad_k))
    base = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)), 1)
    # atpakaļ mazliet "pievelkam", lai nav trekna mala
    base = cv2.erode(base, np.ones((3, 3), np.uint8), 1)

    shift = max(2, int(dil * shift_k))
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    base_shifted = cv2.warpAffine(base, M, (w, h))
    mouth = cv2.max(base, base_shifted)

    mouth = _smooth_mask(mouth, 19)
    return mouth

def _teeth_candidates(img_bgr: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
    """
    Zobu kandidātu karte iekš mutes reģiona ar YCrCb sliekšņiem un adaptīvu statistiku.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    roi = (mouth_mask > 10)
    if not np.any(roi):
        return np.zeros_like(mouth_mask)

    # statistika tikai iekš mutes reģiona
    Y_roi = Y[roi]
    Cr_roi = Cr[roi]

    # Luminance virs vidējā + dispersija; Cr (sarkanums) zem kvantiles
    y_thr = max(140, int(np.clip(np.mean(Y_roi) + 0.6*np.std(Y_roi), 120, 210)))
    cr_thr = int(np.quantile(Cr_roi, 0.55))  # zem šī – mazāk sarkanīgs (mazāk gum/lips)

    cand = (Y >= y_thr) & (Cr <= cr_thr)
    cand = cand.astype(np.uint8) * 255

    # maskējam ar mutes reģionu
    cand = cv2.bitwise_and(cand, mouth_mask)

    # aizpildām caurumus & izlīdzinam
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    cand = cv2.medianBlur(cand, 5)

    return cand

def _apply_whitening(img_bgr: np.ndarray, teeth_mask: np.ndarray, level: int = 6) -> np.ndarray:
    """
    Whiten efekts tikai tur, kur teeth_mask>0.
    level: 1..8
    - paceļ L (LAB) kanālu un samazina b (dzelteno).
    """
    level = int(np.clip(level, 1, 8))
    strength = (level - 1) / 7.0  # 0..1

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # palielinām gaišumu un samazinām dzeltenumu
    L2 = L.astype(np.float32)
    B2 = B.astype(np.float32)

    L2[teeth_mask > 0] = np.clip(L2[teeth_mask > 0] + (22 + 18*strength), 0, 255)
    B2[teeth_mask > 0] = np.clip(B2[teeth_mask > 0] - (18 + 16*strength), 0, 255)

    out = cv2.merge((L2.astype(np.uint8), A, B2.astype(np.uint8)))
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # maiga pāreja (alpha = mask/255, ar nelielu blur)
    alpha = _smooth_mask(teeth_mask, 17).astype(np.float32) / 255.0
    alpha = alpha[..., None]

    blended = (out_bgr.astype(np.float32) * alpha + img_bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return blended

def _find_landmarks(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = FACE_MESH.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return res.multi_face_landmarks[0].landmark

# -----------------------
# API
# -----------------------
@app.route("/whiten", methods=["POST"])
def whiten():
    # -------- input ----------
    if "file" not in request.files:
        return jsonify({"error": "missing file field 'file' (multipart/form-data)"}), 400

    level = request.form.get("level", default="6")
    try:
        level = int(level)
    except Exception:
        level = 6

    try:
        pad_k = float(request.form.get("pad", "0.026"))
        shift_k = float(request.form.get("shift", str(1/3)))
    except Exception:
        pad_k, shift_k = 0.026, (1/3)

    file = request.files["file"]
    buf = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "could not decode image"}), 400

    # -------- landmarks & masks ----------
    lms = _find_landmarks(img_bgr)
    if lms is None:
        # ja neatrod seju, atgriežam oriģinālu (labāk nekā sabojāt)
        _, out_buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return send_file(io.BytesIO(out_buf.tobytes()), mimetype="image/jpeg")

    mouth_mask = _build_mouth_mask(img_bgr, lms, pad_k=pad_k, shift_k=shift_k)
    teeth_mask = _teeth_candidates(img_bgr, mouth_mask)

    # ja kandidātu ļoti maz – neatrisinām ar krāsu, atgriežam oriģinālu
    if cv2.countNonZero(teeth_mask) < 120:
        _, out_buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return send_file(io.BytesIO(out_buf.tobytes()), mimetype="image/jpeg")

    # -------- whitening ----------
    out = _apply_whitening(img_bgr, teeth_mask, level=level)

    # -------- response ----------
    ok, out_buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return send_file(io.BytesIO(out_buf.tobytes()), mimetype="image/jpeg")


@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "msg": "Use POST /whiten with multipart field 'file' and optional 'level', 'pad', 'shift'."})


if __name__ == "__main__":
    # lokālai palaišanai
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
