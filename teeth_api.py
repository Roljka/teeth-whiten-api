import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import traceback
from urllib.request import urlopen
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

# ------------ MediaPipe ------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# Iekšējās lūpas landmarķi (468 shēma)
INNER_LIP_IDX = np.array(
    [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95],
    dtype=np.int32
)

# ------------ TUNING (mutes/zonu ģeometrija) ------------
MOUTH_DILATE_KX_SCALE = 0.003  # sānu “pastiepums”
MOUTH_DILATE_KY_SCALE = 0.008  # vertikālais “pastiepums”
MOUTH_DILATE_ITERS    = 1
MOUTH_EDGE_GUARD      = 10     # atkāpe no lūpas malas (px)
MOUTH_FEATHER_PX      = 13

# Selektori tumšai gaismai/dzeltenumam
ALLOW_DARKER_L = 70
ALLOW_YELLO_B  = 70
SIDE_GROW_PX   = 55
RED_SAT_MIN    = 100

# ------------ Balināšanas līmeņi ------------
# L_gain/L_bias – gaišuma pieaugums (LAB L)
# B_gain/B_bias – dzeltenuma samazinājums (LAB B). Mazāks -> mazāk dzeltens.
# A_neutral     – A kanāla pievilkšana pie 128 (1.0 = neskart)
# blend         – cik daudz jauno zobu ieblendet (0..1)
WHITEN_LEVELS = {
    "tone3":     dict(L_gain=1.04, L_bias=4,  B_gain=0.97, B_bias=-1, A_neutral=0.95, blend=0.70),
    "tone5":     dict(L_gain=1.08, L_bias=7,  B_gain=0.93, B_bias=-2, A_neutral=0.90, blend=0.80),
    "tone8":     dict(L_gain=1.14, L_bias=10, B_gain=0.88, B_bias=-3, A_neutral=0.88, blend=0.86),
    "hollywood": dict(L_gain=1.18, L_bias=13, B_gain=0.84, B_bias=-4, A_neutral=0.86, blend=0.90),
}
LEVEL_ALIASES = {
    "3": "tone3",
    "5": "tone5",
    "8": "tone8",
    "holo": "hollywood",
}

# ------------ Palīgfunkcijas ------------
def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

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
    H, S, _ = cv2.split(hsv)

    m = mouth_mask > 0
    if not np.any(m):
        return np.zeros_like(mouth_mask)

    # CLAHE uz pilnā L, pēc tam iemiksējam mutes zonā
    L_full_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L_eq = L.copy(); L_eq[m] = L_full_eq[m]

    # Sliekšņi no procentīļiem (stabilāk dažādās gaismās)
    Lp = np.percentile(L_eq[m], 55) if np.any(m) else 120
    Bp = np.percentile(B[m], 60)    if np.any(m) else 140
    L_thr = max(40, int(Lp) - ALLOW_DARKER_L)
    B_thr = min(210, int(Bp) + ALLOW_YELLO_B)

    # Sarkanā (lūpas/smaganas) pēc HSV
    red_like = (((H <= 12) | (H >= 170)) & (S > RED_SAT_MIN))

    raw = ((L_eq > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # Atkāpjamies no lūpu malas
    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner_safe = cv2.erode(mouth_mask, guard_k, iterations=1)
    raw = cv2.bitwise_and(raw, inner_safe)

    # Tīrīšana
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)

    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(filled, [c], -1, 255, -1)

    teeth = filled

    # Paplašinām horizontāli (molāri), bet iekš mutes
    if SIDE_GROW_PX > 0:
        kx = SIDE_GROW_PX * 2 + 1
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, 3))
        teeth = cv2.dilate(teeth, grow_k, iterations=1)
        teeth = cv2.bitwise_and(teeth, mouth_mask)

    teeth = cv2.GaussianBlur(teeth, (15, 15), 0)
    return teeth

def _apply_level(img_bgr, teeth_mask, level_key):
    # atlasām profilu
    lvl = level_key.lower()
    lvl = LEVEL_ALIASES.get(lvl, lvl)
    params = WHITEN_LEVELS.get(lvl, WHITEN_LEVELS["tone5"])

    L_gain   = float(params["L_gain"])
    L_bias   = float(params["L_bias"])
    B_gain   = float(params["B_gain"])
    B_bias   = float(params["B_bias"])
    A_neut   = float(params["A_neutral"])
    blend_w  = float(params["blend"])

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0

    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    Lf[m] = np.clip(Lf[m] * L_gain + L_bias, 0, 255)
    Bf[m] = np.clip(Bf[m] * B_gain + B_bias, 0, 255)
    Af[m] = np.clip(128 + (Af[m] - 128) * A_neut, 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # Maigs faktūras izlīdzinājums un kontrolēts blends
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    alpha = (teeth_mask.astype(np.float32) / 255.0)[..., None]
    out_bgr = (img_bgr * (1 - alpha*blend_w) + blur * (alpha*blend_w)).astype(np.uint8)
    return out_bgr

# ------------ Ievades attēla lasīšana ------------
def _read_image_from_request():
    # 1) multipart fails
    if 'file' in request.files:
        f = request.files['file']
    elif 'image' in request.files:
        f = request.files['image']
    else:
        f = None

    # 2) URL (pēc izvēles)
    url = request.form.get('image_url') or request.args.get('image_url')

    try:
        if f:
            pil = Image.open(f.stream)
        elif url:
            # minimāla validācija
            parsed = urlparse(url)
            if not (parsed.scheme in ("http", "https") and parsed.netloc):
                return None, ("invalid image_url", 400)
            with urlopen(url) as resp:
                data = resp.read()
            pil = Image.open(io.BytesIO(data))
        else:
            return None, ("missing file field 'file' or 'image_url'", 400)

        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        return None, (f"cannot open image: {e}", 400)

    img = np.array(pil)[:, :, ::-1]  # RGB->BGR
    return img, None

# ------------ Galvenais darbs ------------
def _teeth_whiten(img_bgr, level="tone5"):
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

    return _apply_level(img_bgr, teeth_mask, level)

# ------------ API ------------
@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        level = request.form.get("level") or request.args.get("level") or "tone5"
        img_bgr, err = _read_image_from_request()
        if err:
            return jsonify({"error": err[0]}), err[1]

        out_bgr = _teeth_whiten(img_bgr, level=level)
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
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
