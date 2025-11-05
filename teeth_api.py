import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import traceback
from urllib.request import urlopen, Request  # URL ielādei bez requests

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

# ---------- Noklusētie TUNING (versija, kas “rullē”) ----------
DEF_EDGE_GUARD   = 9      # px
DEF_FEATHER_PX   = 15
DEF_RED_S_MIN    = 32
DEF_MIN_TOOTH_CC = 80

# Mutes maskas pastiepums
MOUTH_DILATE_KX_SCALE = 0.003  # platums sānos
MOUTH_DILATE_KY_SCALE = 0.008  # mazāka vertikālā dilatācija (mazāk aizskar lūpas)
MOUTH_DILATE_ITERS    = 1
MOUTH_EDGE_GUARD      = 10     # atkāpe no lūpu malas
MOUTH_FEATHER_PX      = 13

# Tumšā gaisma / dzeltenāki zobi (maskas atlasei)
ALLOW_DARKER_L   = 70
ALLOW_YELLO_B    = 70
SIDE_GROW_PX     = 55          # paplašinām uz sāniem
RED_SAT_MIN      = 100         # agresīvi metam ārā sarkano (HSV S slieksnis)

# ---------- Līmeņi (3, 5, 8 toņi, Hollywood) ----------
# Katram līmenim: L_gain, L_bias, B_gain, B_bias, A_neutral_mix, blend_strength
WHITEN_LEVELS = {
    # ~3 toņi
    "tone3":     dict(L_gain=1.05, L_bias=5,  B_gain=0.95, B_bias=-1, A_neutral=0.92, blend=0.75),
    # ~5 toņi
    "tone5":     dict(L_gain=1.08, L_bias=7,  B_gain=0.93, B_bias=-2, A_neutral=0.90, blend=0.80),
    # ~8 toņi
    "tone8":     dict(L_gain=1.12, L_bias=9,  B_gain=0.90, B_bias=-3, A_neutral=0.88, blend=0.85),
    # “Hollywood” – balts, bet ar joprojām dabisku neitralizāciju
    "hollywood": dict(L_gain=1.16, L_bias=12, B_gain=0.86, B_bias=-4, A_neutral=0.86, blend=0.88),
}

# --- palīgi parametru lasīšanai (ja gribēsi kādreiz pielikt query) ---
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

    # anizotropa dilatācija
    side = max(1.0, math.sqrt(area))
    kx = max(3, int(side * MOUTH_DILATE_KX_SCALE)) | 1
    ky = max(3, int(side * MOUTH_DILATE_KY_SCALE)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    mask = cv2.dilate(mask, kernel, iterations=MOUTH_DILATE_ITERS)

    # atkāpe no lūpu malas
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
    Stabils zobu segums:
      - CLAHE mutes zonai
      - procentīļu sliekšņi (L/B) + atļaujam tumšākus/dzeltenākus
      - izmetam sarkanos (HSV)
      - drošības atkāpe no lūpas + morfoloģija
      - horizontāla dilatācija sānu zobiem
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

    # CLAHE uz pilna L, tad iemiksējam mutes zonā
    L_full_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L_eq = L.copy()
    L_eq[m] = L_full_eq[m]

    # Dinamiskie sliekšņi
    Lp = np.percentile(L_eq[m], 55) if np.any(m) else 120
    Bp = np.percentile(B[m], 60)    if np.any(m) else 140
    L_thr = max(40, int(Lp) - ALLOW_DARKER_L)
    B_thr = min(210, int(Bp) + ALLOW_YELLO_B)

    # Sarkanais (lūpas/smaganas)
    red_like = (((H <= 12) | (H >= 170)) & (S > RED_SAT_MIN))

    # Kandidāti
    raw = ((L_eq > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # atkāpe no lūpas
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

    # Pastiepjam uz sāniem
    if SIDE_GROW_PX > 0:
        kx = SIDE_GROW_PX * 2 + 1
        ky = 3
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
        teeth = cv2.dilate(teeth, grow_k, iterations=1)
        teeth = cv2.bitwise_and(teeth, mouth_mask)

    # mīksta mala
    teeth = cv2.GaussianBlur(teeth, (15, 15), 0)
    return teeth

# ---------- Attēla ielāde ----------
def _read_image_from_url(url: str):
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=12) as r:
            data = r.read()
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        img = np.array(pil)[:, :, ::-1]  # RGB->BGR
        return img, None
    except Exception as e:
        return None, (f"cannot fetch image_url: {e}", 400)

def _read_image_from_request():
    """
    Pieņem:
      - multipart 'file' vai 'image'
      - vai 'image_url' (form-data vai query)
    """
    image_url = request.form.get("image_url") or request.args.get("image_url")
    if image_url:
        return _read_image_from_url(image_url)

    f = None
    if 'file' in request.files:
        f = request.files['file']
    elif 'image' in request.files:
        f = request.files['image']

    if not f:
        return None, ("missing file field 'file' or 'image_url'", 400)

    try:
        pil = Image.open(f.stream)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        return None, (f"cannot open image: {e}", 400)

    img = np.array(pil)[:, :, ::-1]  # RGB->BGR
    return img, None

# ---------- Balināšana ----------
def _apply_level_color(lab_img, mask, level_key: str):
    """
    Pielieto krāsas korekciju pēc līmeņa.
    """
    params = WHITEN_LEVELS.get(level_key, WHITEN_LEVELS["tone5"])
    L_gain   = params["L_gain"]
    L_bias   = params["L_bias"]
    B_gain   = params["B_gain"]
    B_bias   = params["B_bias"]
    A_neutral= params["A_neutral"]
    blend    = params["blend"]  # cik daudz jauno krāsu miksējam ar oriģinālu

    L, A, B = cv2.split(lab_img)
    m = mask > 0

    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # maigs gaišums
    Lf[m] = np.clip(Lf[m] * L_gain + L_bias, 0, 255)
    # mazāk dzeltenuma (bet ne līdz zilumam)
    Bf[m] = np.clip(Bf[m] * B_gain + B_bias, 0, 255)
    # nedaudz uz neitrālāku A (mazāk rozā/zilzaļās nobīdes)
    Af[m] = np.clip(128 + (Af[m] - 128) * A_neutral, 0, 255)

    lab_new = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    bgr_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    # maigs faktūras izlīdzinājums un dabiskāks blends
    bgr_new = cv2.bilateralFilter(bgr_new, d=7, sigmaColor=40, sigmaSpace=40)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out_bgr = (lab_to_bgr(lab_img) * (1 - alpha*blend) + bgr_new * (alpha*blend)).astype(np.uint8)
    return out_bgr

def lab_to_bgr(lab_img):
    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

def _teeth_whiten(img_bgr, level_key: str):
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

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    out_bgr = _apply_level_color(lab, teeth_mask, level_key)
    return out_bgr

# ---------- API ----------
@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        img_bgr, err = _read_image_from_request()
        if err:
            return jsonify({"error": err[0]}), err[1]

        # līmeņa izvēle (form vai query): tone3 | tone5 | tone8 | hollywood
        level = (request.form.get("level")
                 or request.args.get("level")
                 or "tone5").strip().lower()
        if level not in WHITEN_LEVELS:
            # pieņem arī 3/5/8/holo kā īsos alias
            if level in {"3", "three"}: level = "tone3"
            elif level in {"5", "five"}: level = "tone5"
            elif level in {"8", "eight"}: level = "tone8"
            elif level in {"holo", "hollywood", "hollywood_smile"}: level = "hollywood"
            else:
                level = "tone5"

        out_bgr = _teeth_whiten(img_bgr, level)
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
    return jsonify({"ok": True, "levels": list(WHITEN_LEVELS.keys())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
