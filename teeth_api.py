import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ---------- MediaPipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# Iekšējās lūpas landmarķi (468 shēmā)
INNER_LIP_IDX = np.array([
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,
    87,178,88,95
], dtype=np.int32)

# ---------- Palīgfunkcijas ----------
def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask, k=11):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask, (k, k), 0)

def _build_mouth_mask(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]
    inner = _landmarks_to_xy(landmarks, w, h, INNER_LIP_IDX)

    area = cv2.contourArea(inner)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    # pamata mute
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    # mazāka dilatācija (neizlīst uz lūpām), mērogota pēc mutes izmēra
    dil = max(6, int(math.sqrt(area) * 0.03))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)), 1)

    # lip-guard: izgriežam šauru joslu gar iekšējo lūpu kontūru
    # (neatstāsim “balinātāju” uz lūpām)
    guard = max(4, int(math.sqrt(area) * 0.015))  # ~1.5% no mutes augstuma
    eroded_for_guard = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard*2+1, guard*2+1)), 1)
    lip_guard = cv2.subtract(mask, eroded_for_guard)          # tikai šaurā mala
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(lip_guard))  # noņemam malu

    # vairs NEPABĪDAM uz leju (shift=0), lai nelīstu smaganās
    mask = _smooth_mask(mask, 17)
    return mask

def _build_teeth_mask(img_bgr, mouth_mask):
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    h, w = mouth_mask.shape
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    m = mouth_mask > 0
    # --- adaptīvi sliekšņi tikai mutes zonai (stabili arī sliktā gaismā) ---
    L_roi = L[m]; B_roi = B[m]
    L_thr, _ = cv2.threshold(L_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    B_thr, _ = cv2.threshold(B_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kandidāti: gaiši un nedzelteni
    cand = ((L > L_thr) & (B < B_thr) & m)

    # gum-cut (smaganas/lūpas): sarkans/rozā un/vai liels A + pietiekama S
    gum_red1 = ((H < 15) | (H > 170)) & (S > 40)
    gum_red2 = (A > 148) & (S > 30)
    gums = (gum_red1 | gum_red2)

    # tumšais tukšums (mute iekšā, ēna/tumsa) – zobiem jābūt vismaz vidēji gaišiem
    dark_void = (L < max(80, int(L_thr) - 10))

    # lip-guard atkārtota drošība – vēl šaurāka mala
    guard = 3
    inner_safe = cv2.erode(mouth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard*2+1, guard*2+1)), 1) > 0

    raw = cand & (~gums) & (~dark_void) & inner_safe

    raw = (raw.astype(np.uint8) * 255)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)

    # aizpildām caurumus, lai nav plankumi
    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(filled, [c], -1, 255, -1)

    teeth_mask = _smooth_mask(filled, 13)

    # kvalitātes kontrole – ja par maz pārklājuma, drošais fallback (mute bez smaganām)
    mouth_area = mouth_mask.sum() / 255.0
    teeth_area = teeth_mask.sum() / 255.0
    if teeth_area / max(mouth_area, 1.0) < 0.30:
        base = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(gums.astype(np.uint8)*255))
        base = cv2.erode(base, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
        teeth_mask = _smooth_mask(base, 13)

    return teeth_mask

def _teeth_whiten(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    teeth_mask = _build_teeth_mask(img_bgr, mouth_mask)

    if np.sum(teeth_mask) == 0:
        # pēdējais glābiņš – balinām konservatīvi mutes zonu bez smaganām
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        gum = (A > 148).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(gum))
        mask = _smooth_mask(mask, 15)
        m = mask > 0
        if np.any(m):
            Lf = L.astype(np.float32); Bf = B.astype(np.float32)
            Lf[m] = np.clip(Lf[m] * 1.12 + 10, 0, 255)
            Bf[m] = np.clip(Bf[m] * 0.86 - 6, 0, 255)
            out = cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)])
            out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
            blur = cv2.bilateralFilter(out_bgr, 7, 40, 40)
            out_bgr[m] = blur[m]
            return out_bgr
        return img_bgr

    # Parastā (labā) līnija
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = teeth_mask > 0
    Lf = L.astype(np.float32); Bf = B.astype(np.float32)
    Lf[m] = np.clip(Lf[m] * 1.15 + 12, 0, 255)
    Bf[m] = np.clip(Bf[m] * 0.82 - 8, 0, 255)
    out = cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    blur = cv2.bilateralFilter(out_bgr, 7, 40, 40)
    out_bgr[m] = blur[m]
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

# ---------- API ----------
@app.route("/whiten", methods=["POST"])
def whiten():
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

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
