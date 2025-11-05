import io, math, traceback
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
INNER_LIP_IDX = np.array(
    [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95],
    dtype=np.int32
)

# ---------- Noklusētie TUNING (maskai) ----------
MOUTH_DILATE_KX_SCALE = 0.003   # platāk uz sāniem (molāri)
MOUTH_DILATE_KY_SCALE = 0.016   # vertikāli (mazāk = mazāk uz lūpām)
MOUTH_DILATE_ITERS    = 1
MOUTH_EDGE_GUARD      = 6       # atkāpe no lūpas malas
MOUTH_FEATHER_PX      = 15

# Stabilitātei tumšās bildēs
ALLOW_DARKER_L   = 60
ALLOW_YELLO_B    = 60
SIDE_GROW_PX     = 40
RED_SAT_MIN      = 25

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
    h, w = img_bgr.shape[:2]
    inner = _landmarks_to_xy(landmarks, w, h, INNER_LIP_IDX)

    area = cv2.contourArea(inner)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    side = max(1.0, math.sqrt(area))
    kx = max(3, int(side * MOUTH_DILATE_KX_SCALE)) | 1
    ky = max(3, int(side * MOUTH_DILATE_KY_SCALE)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    mask = cv2.dilate(mask, kernel, iterations=MOUTH_DILATE_ITERS)

    if MOUTH_EDGE_GUARD > 0:
        g = MOUTH_EDGE_GUARD
        guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g*2+1, g*2+1))
        mask = cv2.erode(mask, guard_k, iterations=1)

    if MOUTH_FEATHER_PX > 0:
        f = MOUTH_FEATHER_PX | 1
        mask = cv2.GaussianBlur(mask, (f, f), 0)

    return mask

def _build_teeth_mask(img_bgr, mouth_mask):
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    m = mouth_mask > 0
    if not np.any(m):
        return np.zeros_like(mouth_mask)

    # CLAHE tikai mutes zonai
    L_full_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L_eq = L.copy()
    L_eq[m] = L_full_eq[m]

    # Dinamiski sliekšņi (procentīļi) + pielaides
    Lp = np.percentile(L_eq[m], 55) if np.any(m) else 120
    Bp = np.percentile(B[m], 60)    if np.any(m) else 140
    L_thr = max(40, int(Lp) - ALLOW_DARKER_L)
    B_thr = min(210, int(Bp) + ALLOW_YELLO_B)

    # Izmet sarkano (smaganas/lūpas) pēc HSV
    red_like = (((H <= 12) | (H >= 170)) & (S > RED_SAT_MIN))

    raw = ((L_eq > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # drošības josla no lūpas malas
    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner_safe = cv2.erode(mouth_mask, guard_k, iterations=1)
    raw = cv2.bitwise_and(raw, inner_safe)

    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(filled, [c], -1, 255, thickness=-1)

    teeth = filled

    # Paplašinām horizontāli (molāriem), bet paliekam mutes zonā
    if SIDE_GROW_PX > 0:
        kx = SIDE_GROW_PX * 2 + 1
        ky = 3
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
        teeth = cv2.dilate(teeth, grow_k, iterations=1)
        teeth = cv2.bitwise_and(teeth, mouth_mask)

    teeth = cv2.GaussianBlur(teeth, (15, 15), 0)
    return teeth

# ---------- Balināšanas līmeņi ----------
# Katram: (L_mult, L_add, B_mult, B_sub, A_towards_128, blend_alpha)
LEVELS = {
    "3":         (1.04,  6, 0.95, 1, 0.94, 0.75),  # 3 toņi
    "5":         (1.06,  8, 0.92, 2, 0.92, 0.80),  # 5 toņi
    "8":         (1.08, 10, 0.88, 3, 0.90, 0.85),  # 8 toņi
    "hollywood": (1.12, 12, 0.82, 5, 0.88, 0.90),  # Holivudas smaids
}

def _apply_whitening(img_bgr, teeth_mask, level_key):
    # drošs defaults
    if level_key not in LEVELS:
        level_key = "5"
    Lm, La, Bm, Bs, Acoef, alpha = LEVELS[level_key]

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0

    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # Gaišums
    Lf[m] = np.clip(Lf[m] * Lm + La, 0, 255)
    # Dzeltenuma samazināšana (bez “zilas plastmasas”)
    Bf[m] = np.clip(Bf[m] * Bm - Bs, 0, 255)
    # Tuvāk neitralam (mazāk rozā/zilzaļais tonis)
    Af[m] = np.clip(128 + (Af[m] - 128) * Acoef, 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # Maigs faktūras izlīdzinājums + dabiskāka sajaukšana
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    a = (teeth_mask.astype(np.float32) / 255.0)[..., None]  # [H,W,1]
    out_bgr = (img_bgr * (1 - a * alpha) + blur * (a * alpha)).astype(np.uint8)
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

def _teeth_whiten(img_bgr, level_key="5"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr
    landmarks = res.multi_face_landmarks[0].landmark

    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    if mouth_mask.sum() == 0:
        return img_bgr

    teeth_mask = _build_teeth_mask(img_bgr, mouth_mask)
    if np.sum(teeth_mask) == 0:
        return img_bgr

    return _apply_whitening(img_bgr, teeth_mask, level_key)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        level = (request.form.get("level") or request.args.get("level") or "").lower()
        if level not in LEVELS:
            level = "5"  # defaults

        img_bgr, err = _read_image_from_request()
        if err:
            return jsonify({"error": err[0]}), err[1]

        out_bgr = _teeth_whiten(img_bgr, level_key=level)

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(out_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": f"processing_failed: {type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
