# teeth_api.py
import io, os
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

TARGET_L = 94.0
FEATHER = 6
KEEP_COMPONENTS = 2

_mp = None
_facemesh = None
OUTER_IDX = [61,146,91,181,84,17,314,405,321,375,291,308]
INNER_IDX = [78,95,88,178,87,14,317,402,318,324,308,415]

def get_facemesh():
    global _mp, _facemesh
    if _facemesh is None:
        import mediapipe as mp
        _mp = mp
        _facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return _mp, _facemesh

def landmarks_to_polygon(lms, idxs, w, h):
    pts = [(int(lms[i].x*w), int(lms[i].y*h)) for i in idxs]
    return np.array(pts, dtype=np.int32)

def fill_poly_mask(img_shape, poly):
    """img_shape = (H, W, C) vai (H, W)"""
    h, w = img_shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [poly], 255)
    return m

def keep_biggest(mask, k=KEEP_COMPONENTS, min_area=120):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = []
    for i in range(1, num):
        areas.append((stats[i, cv2.CC_STAT_AREA], i))
    areas.sort(reverse=True)
    keep = np.zeros_like(mask)
    taken = 0
    for area, idx in areas:
        if area >= min_area:
            keep[lab==idx] = 255
            taken += 1
        if taken >= k:
            break
    return keep

def feather_mask(mask, r=FEATHER):
    if r <= 0:
        return (mask>0).astype(np.float32)
    dist = cv2.distanceTransform((mask==0).astype(np.uint8), cv2.DIST_L2, 3)
    band = np.clip(dist, 0, r)/float(r)
    soft = np.clip((mask>0).astype(np.float32)*(1.0-(1.0-band)), 0, 1)
    return soft

def adaptive_teeth_mask(img_bgr, inner_mask):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    roi = inner_mask>0
    if roi.sum() < 50:
        return np.zeros_like(L, dtype=np.float32)

    Lr = L[roi].astype(np.float32)
    Ar = A[roi].astype(np.float32)
    Br = B[roi].astype(np.float32)
    Sr = S[roi].astype(np.float32)
    Vr = V[roi].astype(np.float32)

    pL70 = np.percentile(Lr, 70)
    pL55 = np.percentile(Lr, 55)
    pS60 = np.percentile(Sr, 60)
    pS70 = np.percentile(Sr, 70)
    a_med = np.median(Ar)
    b_med = np.median(Br)

    base = (
        (L.astype(np.float32) >= pL70) &
        (S.astype(np.float32) <= pS60) &
        (A.astype(np.float32) <= a_med + 6.0) &
        (B.astype(np.float32) <= b_med + 10.0)
    )
    base = base & roi

    if base.sum() < 0.02*roi.sum():
        base = (
            (L.astype(np.float32) >= pL55) &
            (S.astype(np.float32) <= pS70) &
            (A.astype(np.float32) <= a_med + 8.0) &
            (B.astype(np.float32) <= b_med + 14.0)
        )
        base = base & roi

    base = base.astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, k, iterations=1)
    base = cv2.dilate(base, k, iterations=1)
    base = keep_biggest(base, k=KEEP_COMPONENTS, min_area=max(80, roi.sum()//500))
    soft = feather_mask(base, r=FEATHER)
    return soft

def whiten_with_mask(img_bgr, soft_mask):
    if soft_mask.max() <= 0:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32)
    gain = (TARGET_L - Lf); gain[gain < 0] = 0
    L_new = np.clip(Lf + gain * soft_mask, 0, TARGET_L).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L_new, A, B]), cv2.COLOR_LAB2BGR)
    alpha3 = np.dstack([soft_mask, soft_mask, soft_mask]).astype(np.float32)
    blended = (alpha3*out + (1.0-alpha3)*img_bgr).astype(np.uint8)
    return blended

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "Teeth Whitening API – adaptive teeth-only"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    data = request.files["file"].read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Bad image"}), 400

    mp, fm = get_facemesh()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")

    h, w = img.shape[:2]
    lms = res.multi_face_landmarks[0].landmark
    inner_poly = landmarks_to_polygon(lms, INNER_IDX, w, h)
    inner_mask = fill_poly_mask(img.shape, inner_poly)

    soft = adaptive_teeth_mask(img, inner_mask)
    out = whiten_with_mask(img, soft)

    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg", download_name="whitened.jpg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
