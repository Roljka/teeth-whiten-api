# teeth_api.py

import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import traceback
import requests

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

# ---------- Līmeņu iestatījumi ----------
# Katrs līmenis nosaka:
#  - L_GAIN, L_ADD: cik celt gaišumu (LAB L)
#  - B_MUL, B_ADD: cik mazināt dzeltenumu (LAB B)
#  - A_BLEND: cik ļoti tuvot A pie neitrālā 128 (0..1; 0=nelabot, 1=pilnīgi uz 128)
#  - ALPHA: gala sajaukums ar oriģinālu (dabiskumam). 0.85 nozīmē 85% “jaunais”.
LEVELS = {
    "3tones":   {"L_GAIN": 1.05, "L_ADD": 4,  "B_MUL": 0.95, "B_ADD": -1, "A_BLEND": 0.08, "ALPHA": 0.70},
    "5tones":   {"L_GAIN": 1.08, "L_ADD": 7,  "B_MUL": 0.92, "B_ADD": -2, "A_BLEND": 0.10, "ALPHA": 0.78},
    "8tones":   {"L_GAIN": 1.12, "L_ADD": 10, "B_MUL": 0.88, "B_ADD": -3, "A_BLEND": 0.12, "ALPHA": 0.85},
    "hollywood":{"L_GAIN": 1.18, "L_ADD": 14, "B_MUL": 0.82, "B_ADD": -4, "A_BLEND": 0.16, "ALPHA": 0.90},
}

DEFAULT_LEVEL = "5tones"

# ---------- Noklusētie TUNING ----------
DEF_EDGE_GUARD   = 6
DEF_FEATHER_PX   = 15

# Mutes maskas pastiepums (anizotropa dilatācija)
MOUTH_DILATE_KX_SCALE = 0.003
MOUTH_DILATE_KY_SCALE = 0.016
MOUTH_DILATE_ITERS    = 1
MOUTH_EDGE_GUARD      = 6
MOUTH_FEATHER_PX      = 15

# Tumšā gaisma / dzeltenāki zobi
ALLOW_DARKER_L   = 60
ALLOW_YELLO_B    = 60
SIDE_GROW_PX     = 40
RED_SAT_MIN      = 25

def _getf(name, default):
    v = request.args.get(name)
    if v is None: 
        return float(default)
    try:
        return float(v)
    except:
        return float(default)

def _geti(name, default):
    v = request.args.get(name)
    if v is None: 
        return int(default)
    try:
        return int(v)
    except:
        return int(default)

def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask, k=11):
    k = int(k)
    if k < 1:
        k = 1
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

    # anizotropa dilatācija ar TUNING parametriem
    side = max(1.0, math.sqrt(area))
    kx = max(3, int(side * MOUTH_DILATE_KX_SCALE)) | 1
    ky = max(3, int(side * MOUTH_DILATE_KY_SCALE)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    mask = cv2.dilate(mask, kernel, iterations=MOUTH_DILATE_ITERS)

    # drošības josla no lūpu malas
    if MOUTH_EDGE_GUARD > 0:
        g = MOUTH_EDGE_GUARD
        guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g*2+1, g*2+1))
        mask = cv2.erode(mask, guard_k, iterations=1)

    # mīksta mala
    if MOUTH_FEATHER_PX > 0:
        f = MOUTH_FEATHER_PX | 1
        mask = cv2.GaussianBlur(mask, (f, f), 0)

    return mask

def _build_teeth_mask(img_bgr, mouth_mask):
    """
    Stabils zobu segums arī tumšākās bildēs:
      - CLAHE tikai mutes zonai (caur pilna L ceļu)
      - sliekšņi no procentīļiem
      - atļaujam tumšākus/dzeltenākus zobus
      - izmetam sarkanos (smaganas/lūpas) pēc HSV
      - horizontāla “pastiepšana” uz sāniem
    """
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    h, w = mouth_mask.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    m = mouth_mask > 0
    if not np.any(m):
        return np.zeros_like(mouth_mask)

    # 1) CLAHE – uz pilnā L, tad ieliekam mutes zonā
    L_full_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L_eq = L.copy()
    L_eq[m] = L_full_eq[m]

    # 2) Dinamiski sliekšņi no procentīļiem mutes zonā
    Lp = np.percentile(L_eq[m], 55) if np.any(m) else 120
    Bp = np.percentile(B[m], 60)    if np.any(m) else 140
    L_thr = max(40, int(Lp) - ALLOW_DARKER_L)
    B_thr = min(210, int(Bp) + ALLOW_YELLO_B)

    # 3) Sarkanais (smaganas/lūpas) pēc HSV
    red_like = (((H <= 12) | (H >= 170)) & (S > RED_SAT_MIN))

    # 4) Kandidāti
    raw = ((L_eq > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # drošības malas atkāpe no lūpas
    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner_safe = cv2.erode(mouth_mask, guard_k, iterations=1)
    raw = cv2.bitwise_and(raw, inner_safe)

    # morfoloģija
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    # aizpildām caurumus
    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(filled, [c], -1, 255, thickness=-1)

    teeth = filled

    # 5) Paplašinām horizontāli (uz sāniem), bet paliekam mutes zonā
    if SIDE_GROW_PX > 0:
        kx = SIDE_GROW_PX * 2 + 1
        ky = 3
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
        teeth = cv2.dilate(teeth, grow_k, iterations=1)
        teeth = cv2.bitwise_and(teeth, mouth_mask)

    # mīksta mala
    teeth = cv2.GaussianBlur(teeth, (15, 15), 0)
    return teeth

def _apply_level_whitening(img_bgr, teeth_mask, level_key):
    """Piemēro LAB korekcijas atbilstoši izvēlētajam līmenim + dabisku sajaukumu."""
    params = LEVELS.get(level_key, LEVELS[DEFAULT_LEVEL])

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0

    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # Gaišums (L): multiplikatīvs + aditīvs pieaugums
    Lf[m] = np.clip(Lf[m] * params["L_GAIN"] + params["L_ADD"], 0, 255)

    # Dzeltenuma mazinājums (B)
    Bf[m] = np.clip(Bf[m] * params["B_MUL"] + params["B_ADD"], 0, 255)

    # Nedaudz tuvāk neitralam A=128 (ne pārāk, lai nesanāk “zili zobi”)
    if params["A_BLEND"] > 0:
        Af[m] = np.clip(128 + (Af[m] - 128) * (1.0 - params["A_BLEND"]), 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # Maigs faktūras izlīdzinājums + dabiskāks blends ar oriģinālu
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    alpha = (teeth_mask.astype(np.float32) / 255.0)[..., None] * float(params["ALPHA"])
    out_bgr = (img_bgr * (1 - alpha) + blur * alpha).astype(np.uint8)
    return out_bgr

def _teeth_whiten(img_bgr, level_key):
    # MediaPipe
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

    return _apply_level_whitening(img_bgr, teeth_mask, level_key)

def _pil_from_bytes(b):
    pil = Image.open(io.BytesIO(b))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return pil

def _read_image_from_request():
    """
    Pieņem:
      - multipart 'file' vai 'image'
      - vai form-data 'image_url' (server-side fetch)
    Atgriež: (BGR ndarray, None) vai (None, (error_msg, status))
    """
    try:
        if 'file' in request.files:
            f = request.files['file']
            pil = Image.open(f.stream)
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            img = np.array(pil)[:, :, ::-1]  # RGB->BGR
            return img, None

        if 'image' in request.files:
            f = request.files['image']
            pil = Image.open(f.stream)
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            img = np.array(pil)[:, :, ::-1]
            return img, None

        image_url = request.form.get("image_url") or request.args.get("image_url")
        if image_url:
            resp = requests.get(image_url, timeout=10)
            if resp.status_code != 200:
                return None, (f"cannot fetch image_url (status {resp.status_code})", 400)
            ctype = resp.headers.get("Content-Type", "")
            if "image" not in ctype:
                return None, (f"image_url returned non-image content-type: {ctype}", 400)
            pil = _pil_from_bytes(resp.content)
            img = np.array(pil)[:, :, ::-1]
            return img, None

        return None, ("missing file field 'file' (multipart/form-data) or 'image_url'", 400)
    except Exception as e:
        return None, (f"cannot open image: {e}", 400)

# ---------- API ----------
@app.route("/levels", methods=["GET"])
def list_levels():
    return jsonify({
        "default": DEFAULT_LEVEL,
        "levels": list(LEVELS.keys()),
        "params": LEVELS
    })

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        # level var nākt kā form-data vai query
        level_key = request.form.get("level") or request.args.get("level") or DEFAULT_LEVEL
        if level_key not in LEVELS:
            level_key = DEFAULT_LEVEL

        img_bgr, err = _read_image_from_request()
        if err:
            return jsonify({"error": err[0]}), err[1]

        out_bgr = _teeth_whiten(img_bgr, level_key)

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(out_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({
            "error": f"processing_failed: {type(e).__name__}: {e}",
            "trace": traceback.format_exc()
        }), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "levels": list(LEVELS.keys()), "default": DEFAULT_LEVEL})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
