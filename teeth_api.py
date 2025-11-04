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

# ---------- Noklusētie TUNING ----------
DEF_DIL_H_SCALE = 0.065   # horizontāli (nedaudz plašāk, bet droši)
DEF_DIL_V_SCALE = 0.020   # vertikāli (mazāk, lai neuzkāpj uz lūpām)
DEF_EDGE_GUARD   = 7      # px (lielāka sargjosla no lūpas)
DEF_FEATHER_PX   = 15
DEF_A_MAX        = 138    # stingrāk pret smaganām (rozā/sarkans)
DEF_RED_H_LOW    = 15     # šaurāks “red” logs
DEF_RED_H_HIGH   = 165
DEF_RED_S_MIN    = 42     # vajag sarkanāku, lai izkristu
DEF_L_DELTA      = -14    # atļaujam nedaudz tumšākus
DEF_B_DELTA      = +26    # atļaujam nedaudz dzeltenākus
DEF_MIN_TOOTH_CC = 80

# Mutes maskas pastiepums
MOUTH_DILATE_KX_SCALE = 0.016  # platāk uz sāniem (drošāk par SIDE_GROW)
MOUTH_DILATE_KY_SCALE = 0.014  # mazāk vertikāli (neuzkāpj uz lūpām)
MOUTH_DILATE_ITERS    = 2
MOUTH_EDGE_GUARD      = 7
MOUTH_FEATHER_PX      = 15

# Tumšā gaisma / dzeltenāki zobi
ALLOW_DARKER_L   = 45    # pietiek zobiem tumšā gaismā, bet nepaceļ smaganas
ALLOW_YELLO_B    = 40    # atļaujam dzeltenīgākus bez gum-bleed
SIDE_GROW_PX     = 24    # mazāk agresīva “izplešana” (mazāk ies krāsā uz lūpām)
RED_SAT_MIN      = 40    # konsekventi ar DEF_RED_S_MIN, lai sarkanais krīt ārā

def _getf(name, default):
    v = request.args.get(name)
    if v is None: return float(default)
    try: return float(v)
    except: return float(default)

def _geti(name, default):
    v = request.args.get(name)
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
      - CLAHE tikai mutes zonai (caur pilna L ceļu, lai nebūtu 1D kļūdas)
      - sliekšņi no procentīļiem
      - atļaujam tumšākus/dzeltenākus zobus
      - izmetam sarkanos (smaganas/lūpas)
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

    # 1) CLAHE – APPLY UZ PILNĀ L (2D), PĒC TAM IEBLENDĒ MUTES ZONĀ
    L_full_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L_eq = L.copy()
    L_eq[m] = L_full_eq[m]

    # 2) Dinamiski sliekšņi no procentīļiem mutes zonā
    # (sargi, ja reģions sanāk tukšs)
    if not np.any(m):
        return np.zeros_like(mouth_mask)

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

def _teeth_whiten(img_bgr):
    _ = _geti("feather", DEF_FEATHER_PX)  # hook nākotnei

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
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0

    Lf = L.astype(np.float32)
    Bf = B.astype(np.float32)
    Lf[m] = np.clip(Lf[m] * 1.10 + 10, 0, 255)
    Bf[m] = np.clip(Bf[m] * 0.84 - 6, 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # viegls faktūras izlīdzinājums tikai zobu zonā
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
    try:
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
    except Exception as e:
        # palīdzēs ātri saprast, ja vēl kas nobrūk prodā
        return jsonify({"error": f"processing_failed: {type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
