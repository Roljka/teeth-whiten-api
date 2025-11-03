import io
import os
import cv2
import numpy as np
import threading
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------
# Mediapipe (viena instance + lock; MP nav thread-safe)
# -----------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)
_face_lock = threading.Lock()

# MP dokumentos ir divas mutes “loki” – ārējais un iekšējais
MOUTH_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
]
MOUTH_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191
]

# -----------------------------------------------------------
# Palīgfunkcijas
# -----------------------------------------------------------
def load_image_fix_orientation(file_storage, max_side=None) -> np.ndarray:
    """Ielādē attēlu ar EXIF orientācijas labošanas atbalstu.
       Ja max_side ir None -> ne-mērojam, saglabājam oriģinālo izmēru.
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if max_side is not None:
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def enhance_for_detection(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

def poly_from_landmarks(h, w, landmarks, indices):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    return pts

def mask_from_poly(h, w, pts):
    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask

def build_floor(mask: np.ndarray):
    """atrodam mutes apakšējo malu pa x, lai nogrieztu apakšlūpu"""
    h, w = mask.shape[:2]
    floor = np.full(w, -1, dtype=np.int32)
    ys, xs = np.where(mask > 0)
    for x in range(w):
        col = ys[xs == x]
        if col.size > 0:
            floor[x] = col.max()
    return floor

def cut_below(mask: np.ndarray, floor: np.ndarray, lift_px: int) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    for x in range(w):
        y = floor[x]
        if y >= 0:
            out[y + lift_px:h, x] = 0
    return out

def gums_mask(bgr: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
    """atrodam košās rozā/sarkanas vietas mutē – smaganas/lūpu iekšējā daļa"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)

    red = (((H <= 12) | (H >= 170)) & (S > 35))
    pink = (A > 156)

    gum = np.zeros_like(mouth_mask)
    gum[(mouth_mask > 0) & (red | pink)] = 255

    gum = cv2.morphologyEx(
        gum,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    gum = cv2.erode(gum, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    return gum

# -----------------------------------------------------------
# Zobu maska
# -----------------------------------------------------------
def build_teeth_mask(bgr: np.ndarray, landmarks) -> np.ndarray:
    h, w = bgr.shape[:2]

    outer_pts = poly_from_landmarks(h, w, landmarks, MOUTH_OUTER)
    inner_pts = poly_from_landmarks(h, w, landmarks, MOUTH_INNER)

    outer_mask = mask_from_poly(h, w, outer_pts)  # lūpas+mute
    inner_mask = mask_from_poly(h, w, inner_pts)  # zobu “caurums”

    if np.count_nonzero(inner_mask) == 0:
        return np.zeros((h, w), np.uint8)

    inner_wide = cv2.dilate(
        inner_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (33, 5)),
        iterations=1,
    )

    lips_only = cv2.subtract(outer_mask, cv2.dilate(inner_mask, None, iterations=1))

    floor = build_floor(outer_mask)
    inner_wide = cut_below(inner_wide, floor, lift_px=2)
    lips_only = cut_below(lips_only, floor, lift_px=0)

    gum = gums_mask(bgr, inner_wide)
    teeth_mask = cv2.subtract(inner_wide, gum)
    teeth_mask = cv2.subtract(teeth_mask, lips_only)

    filled = np.count_nonzero(teeth_mask)
    mouth_area = np.count_nonzero(inner_wide)
    if mouth_area > 0 and filled / mouth_area < 0.28:
        teeth_mask = cv2.subtract(inner_wide, lips_only)
        teeth_mask = cv2.subtract(teeth_mask, gum)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, k3, iterations=1)

    return teeth_mask

# -----------------------------------------------------------
# Balināšana
# -----------------------------------------------------------
def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 10, b_shift: int = 22) -> np.ndarray:
    if np.count_nonzero(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask = teeth_mask > 0

    Ln = L.astype(np.int16)
    Bn = B.astype(np.int16)
    Ln[mask] = np.clip(Ln[mask] + l_gain, 0, 255)
    Bn[mask] = np.clip(Bn[mask] - b_shift, 0, 255)

    out = cv2.cvtColor(
        cv2.merge([Ln.astype(np.uint8), A, Bn.astype(np.uint8)]),
        cv2.COLOR_LAB2BGR
    )
    return out

# -----------------------------------------------------------
# API
# -----------------------------------------------------------
@app.route("/health")
def health():
    return jsonify(ok=True)

def _get_upload_file():
    # Pieņem "file" vai "image"
    if "file" in request.files:
        return request.files["file"]
    if "image" in request.files:
        return request.files["image"]
    return None

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        up = _get_upload_file()
        if up is None:
            return jsonify(error="file missing (expected 'file' or 'image')"), 400

        # Nemērojam lejup – paturam oriģinālo izmēru
        bgr = load_image_fix_orientation(up, max_side=None)

        # MediaPipe process ar lock (MP nav thread-safe)
        det_img = enhance_for_detection(bgr)
        with _face_lock:
            res = face_mesh.process(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        teeth_mask = build_teeth_mask(det_img, landmarks)

        # Parametrizējami “stiprumi”
        l_gain = int(request.form.get("l_gain", 10))
        b_shift = int(request.form.get("b_shift", 22))

        out = whiten_only_teeth(bgr, teeth_mask, l_gain=l_gain, b_shift=b_shift)

        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify(error="encode failed"), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="whitened.jpg"
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/debug_mask", methods=["POST"])
def debug_mask():
    """Atgriež PNG masku ātrai QA (balts=balināmā zona)."""
    try:
        up = _get_upload_file()
        if up is None:
            return jsonify(error="file missing (expected 'file' or 'image')"), 400

        bgr = load_image_fix_orientation(up, max_side=None)
        det_img = enhance_for_detection(bgr)
        with _face_lock:
            res = face_mesh.process(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        teeth_mask = build_teeth_mask(det_img, landmarks)

        png = cv2.imencode(".png", teeth_mask)[1].tobytes()
        return send_file(
            io.BytesIO(png),
            mimetype="image/png",
            as_attachment=False,
            download_name="mask.png"
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
