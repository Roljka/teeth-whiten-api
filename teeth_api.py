import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ───────────────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh (stabils, viegls)
# ───────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,       # labāka lūpu iekšpuse
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Indeksi lūpu kontūrām (MediaPipe FaceMesh 468 punkti)
OUTER_LIPS = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
INNER_LIPS = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146]

# ───────────────────────────────────────────────────────────────────────────────
# Palīgfunkcijas
# ───────────────────────────────────────────────────────────────────────────────
def _landmarks_bgr(image_bgr):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([[int(l.x * w), int(l.y * h)] for l in lms], dtype=np.int32)
    return pts

def _build_mouth_mask(img_bgr, lms, pad_k=0.016, shift_k=1/3):
    """
    Veido mutes masku no INNER_LIPS, ar nelielu dilatāciju (pad_k) un nelielu vertikālu
    izstiepumu uz augšu un leju (shift_k no maskas augstuma), lai vienmēr paņemtu
    zobu augšas/apakšas, bet neskartu lūpas.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if lms is None or len(lms) < 468:
        return mask

    inner = lms[INNER_LIPS]
    if len(inner) < 3:
        return mask

    cv2.fillPoly(mask, [inner.astype(np.int32)], 255)

    # nedaudz izstiepjam vertikāli (bet mazāk nekā iepriekš)
    ys = inner[:, 1]
    y_min, y_max = int(ys.min()), int(ys.max())
    band_h = y_max - y_min
    up = max(0, y_min - int(band_h * shift_k * 0.45))
    dn = min(h, y_max + int(band_h * shift_k * 0.45))
    mouth_band = np.zeros_like(mask)
    mouth_band[up:dn, :] = 255
    mask = cv2.bitwise_and(mask, mouth_band)

    # neliela dilatācija – “tighter” padding
    pad = max(1, int(round(h * pad_k)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # feather – lai malas ir maigākas
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=pad/2.0 + 0.5)
    _, mask = cv2.threshold(mask, 16, 255, cv2.THRESH_BINARY)
    return mask

def _teeth_candidates(img_bgr, mouth_mask):
    """
    Kandidātu karte zobiem tikai mutes zonā – Lab + HSV heuristikas.
    """
    h, w = img_bgr.shape[:2]
    if cv2.countNonZero(mouth_mask) == 0:
        return np.zeros((h, w), np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # tikai mutē
    m = mouth_mask > 0
    Lm = L[m]

    # dinamiskais slieksnis pēc mutes L vidējā
    thr_L = max(128, int(np.clip(Lm.mean() + 0.35 * Lm.std(), 120, 225)))

    cand = (L >= thr_L) & (S <= 105)  # gaiši + zema piesātinātība
    teeth = np.zeros_like(L, dtype=np.uint8)
    teeth[cand & m] = 255

    # glābšana, ja ēnas: paaugstinām vēl nedaudz reģionu ap already-detected
    if cv2.countNonZero(teeth) < 120:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        teeth = cv2.dilate(teeth, kernel, 1)
        teeth = cv2.bitwise_and(teeth, mouth_mask)

    # tīrīšana
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN, kernel, 1)
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE, kernel, 2)

    # maiga plume
    teeth = cv2.GaussianBlur(teeth, (0, 0), 1.0)
    _, teeth = cv2.threshold(teeth, 32, 255, cv2.THRESH_BINARY)
    return teeth

def _apply_whitening(img_bgr, mask_bin, level=6):
    """
    Balināšana Lab telpā – palielinām L un pietuvinām a/b pie 0 (ne rozā, ne dzeltenu).
    level: 1..8
    """
    level = int(np.clip(level, 1, 8))
    strength = 0.09 + (level - 1) * 0.035  # maigs diapazons
    strength = float(np.clip(strength, 0.05, 0.35))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    # feather mask
    feather = cv2.GaussianBlur(mask_bin, (0, 0), 1.2)
    alpha = (feather.astype(np.float32) / 255.0) * strength
    alpha3 = cv2.merge([alpha, alpha, alpha])

    # L uz augšu, a/b uz 128 (neitrāls)
    Lf = L.astype(np.float32)
    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    L_new = np.clip(Lf + alpha * 55.0, 0, 255)
    a_new = np.clip(af + (128.0 - af) * alpha, 0, 255)
    b_new = np.clip(bf + (128.0 - bf) * alpha, 0, 255)

    out_lab = cv2.merge([L_new.astype(np.uint8),
                         a_new.astype(np.uint8),
                         b_new.astype(np.uint8)])
    out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# API
# ───────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

def _get_upload_bytes():
    # pieņemam vairākus lauku nosaukumus; atbalstām arī base64 stringu
    possible = ["file", "image", "photo", "upload", "picture"]
    f = None
    for k in possible:
        if k in request.files:
            f = request.files[k]
            break
    if f:
        return f.read()

    # base64 form field
    import base64, re
    for k in possible:
        val = request.form.get(k)
        if not val:
            continue
        m = re.match(r"^data:image/(png|jpe?g|webp);base64,(.*)$", val, re.I)
        try:
            return base64.b64decode(m.group(2) if m else val)
        except Exception:
            pass
    return None

@app.route("/whiten", methods=["POST", "OPTIONS"])
def whiten():
    if request.method == "OPTIONS":
        return ("", 204)

    # parametri (pēc vajadzības)
    level = request.form.get("level", "6")
    try:
        level = int(level)
    except Exception:
        level = 6

    # “tighter” maska – mazāks padding
    try:
        pad = float(request.form.get("pad", "0.016"))
    except Exception:
        pad = 0.016
    try:
        shift_k = float(request.form.get("shift", str(1/3)))
    except Exception:
        shift_k = 1/3

    raw = _get_upload_bytes()
    if not raw:
        return jsonify({"error": "missing file field 'file' (multipart/form-data)"}), 400

    buf = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    lms = _landmarks_bgr(img)
    if lms is None:
        # ja nav sejas – vienkārši atgriežam oriģinālu
        ok, out_buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return send_file(io.BytesIO(out_buf.tobytes()), mimetype="image/jpeg")

    mouth_mask = _build_mouth_mask(img, lms, pad_k=pad, shift_k=shift_k)
    teeth_mask = _teeth_candidates(img, mouth_mask)

    if cv2.countNonZero(teeth_mask) < 100:
        ok, out_buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return send_file(io.BytesIO(out_buf.tobytes()), mimetype="image/jpeg")

    out = _apply_whitening(img, teeth_mask, level=level)
    ok, out_buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    return send_file(io.BytesIO(out_buf.tobytes()), mimetype="image/jpeg")
