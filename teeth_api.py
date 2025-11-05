import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import traceback

app = Flask(__name__)
CORS(app)

# ========= DEMO ATTĒLS (NEMAINI, JA VIETĀ) =========
DEMO_IMAGE_URL = "https://woocommerce-181503-5955573.cloudwaysapps.com/wp-content/uploads/2025/11/sieviete-ar-dzelteniem-zobiem.webp"

# ========= BALINĀŠANAS LĪMEŅI (KĀ Bija “labajā” versijā, + skalēti) =========
# Apraksts:
#  l_gain/l_offset – cik stipri ceļam L (gaišumu)
#  b_scale/b_offset – cik mazinām B (dzeltenumu) bez zilguma
#  a_mix – cik tuvinām A kanālu pie neitrāla (128) (0.90 = maigi)
#  alpha – cik daudz no “jaunā” likt uz gala bildes (maskas zonā)
LEVELS = {
    "tone3":     {"l_gain": 1.06, "l_offset": 6,  "b_scale": 0.92, "b_offset": -2, "a_mix": 0.90, "alpha": 0.78},
    "tone5":     {"l_gain": 1.09, "l_offset": 8,  "b_scale": 0.90, "b_offset": -3, "a_mix": 0.88, "alpha": 0.82},
    "tone8":     {"l_gain": 1.12, "l_offset": 10, "b_scale": 0.88, "b_offset": -4, "a_mix": 0.86, "alpha": 0.86},
    "hollywood": {"l_gain": 1.18, "l_offset": 12, "b_scale": 0.85, "b_offset": -6, "a_mix": 0.85, "alpha": 0.90},
}

# ========= MediaPipe =========
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

# ========= TUNING (stabilā maska, kas strādāja) =========
DEF_FEATHER_PX   = 15
DEF_MIN_TOOTH_CC = 80

# Mutes maskas pastiepums (stabilā versija)
MOUTH_DILATE_KX_SCALE = 0.003   # horizontāli (uz sānu zobiem)
MOUTH_DILATE_KY_SCALE = 0.016   # vertikāli (mazāk, lai neuzkāpj uz lūpām)
MOUTH_DILATE_ITERS    = 1
MOUTH_EDGE_GUARD      = 6       # atkāpe no lūpu malas
MOUTH_FEATHER_PX      = 15

# Tumšā gaisma / dzeltenāki zobi (stabilā versija)
ALLOW_DARKER_L   = 60
ALLOW_YELLO_B    = 60
SIDE_GROW_PX     = 40
RED_SAT_MIN      = 25  # sarkanā piesātinājuma min (HSV), lai mazāk ķertu lūpas

def _get_level_params():
    lvl = (request.args.get("level") or request.form.get("level") or "tone5").strip().lower()
    return LEVELS.get(lvl, LEVELS["tone5"])

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

    # anizotropa dilatācija (stabila)
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
      - CLAHE mutes zonai
      - Dinamiski sliekšņi no procentīļiem
      - Pielaižam tumšākus/dzeltenākus zobus
      - Sarkano (lūpas/smaganas) apgriežam pēc HSV (maigi, lai neatgrieztos zilums)
      - Horizontāls pastiepums līdz sānu zobiem
    """
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    m = mouth_mask > 0
    if not np.any(m):
        return np.zeros_like(mouth_mask)

    # 1) CLAHE pilnam L, tad iemiksējam mutes zonā
    L_full_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L_eq = L.copy()
    L_eq[m] = L_full_eq[m]

    # 2) Dinamiskie sliekšņi
    Lp = np.percentile(L_eq[m], 55) if np.any(m) else 120
    Bp = np.percentile(B[m],  60)   if np.any(m) else 140
    L_thr = max(40, int(Lp) - ALLOW_DARKER_L)
    B_thr = min(210, int(Bp) + ALLOW_YELLO_B)

    # 3) Sarkanais (maigi)
    red_like = (((H <= 12) | (H >= 170)) & (S > RED_SAT_MIN))

    # 4) Kandidāti
    raw = ((L_eq > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # drošības atkāpe no lūpām
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

    # 5) Pastiepjam horizontāli (paliekot mutes zonā)
    if SIDE_GROW_PX > 0:
        kx = SIDE_GROW_PX * 2 + 1
        ky = 3
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
        teeth = cv2.dilate(teeth, grow_k, iterations=1)
        teeth = cv2.bitwise_and(teeth, mouth_mask)

    # mīksta mala
    teeth = cv2.GaussianBlur(teeth, (15, 15), 0)
    return teeth

def _apply_level_whitening(img_bgr, teeth_mask, params):
    # Natural white ar līmeņu parametriem
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0

    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # Gaišums
    Lf[m] = np.clip(Lf[m] * params["l_gain"] + params["l_offset"], 0, 255)
    # Dzeltenuma samazināšana (bez zilguma)
    Bf[m] = np.clip(Bf[m] * params["b_scale"] + params["b_offset"], 0, 255)
    # Neitralizācija A kanālam
    Af[m] = np.clip(128 + (Af[m] - 128) * params["a_mix"], 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # Maigs faktūras izlīdzinājums un dabiskāks blends
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    alpha = (teeth_mask.astype(np.float32) / 255.0)[..., None] * float(params["alpha"])
    out_bgr = (img_bgr * (1 - alpha) + blur * alpha).astype(np.uint8)
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

def _teeth_whiten(img_bgr, params):
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

    out_bgr = _apply_level_whitening(img_bgr, teeth_mask, params)
    return out_bgr

# ================== API ==================
@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        params = _get_level_params()
        img_bgr, err = _read_image_from_request()
        if err:
            return jsonify({"error": err[0]}), err[1]

        out_bgr = _teeth_whiten(img_bgr, params)

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(out_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": f"processing_failed: {type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}), 500

@app.route("/demo", methods=["GET"])
def demo():
    # Frontends var paņemt šo URL un ielādēt attēlu bez backend proxī
    return jsonify({"url": DEMO_IMAGE_URL})

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
