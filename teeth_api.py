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

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    # Mazāka paplašināšana; NEbīdam uz leju (mazāk lūpu paķeršanas)
    dil = max(8, int(math.sqrt(area) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask = _smooth_mask(mask, 17)
    return mask

def _build_teeth_mask(img_bgr, mouth_mask):
    """
    Cieša un nepārtraukta zobu maska mutes iekšienē, izmantojot adaptīvus
    (Otsu) sliekšņus L/B kanāliem. Smaganas/lūpas izmetam pēc HSV “sarkanā”.
    Ja rezultāts ir fragmentēts/mazs, atgriež drošu fallback masku.
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

    # --- Otsu sliekšņi tikai mutes zonā ---
    L_roi = L[m]; B_roi = B[m]
    # (neliec blur pa 2D – ROI jau ir 1D; Otsu strādās uz histogrammas)
    L_thr, _ = cv2.threshold(L_roi.astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    B_thr, _ = cv2.threshold(B_roi.astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # HSV sarkanais (lūpas/smaganas): H 0..12 vai 170..180 un pietiekams S
    red_like = (((H <= 12) | (H >= 170)) & (S > 30))

    # Zobi: gaišāki (L > L_thr), mazāk dzelteni (B < B_thr), ne-sarkani, mutes zonā
    raw = ((L > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # Edge guard: novācam iekšējās malas joslu, lai neskar lūpu malu
    EDGE_GUARD = 5
    k_guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_safe = cv2.erode(mouth_mask, k_guard, iterations=1)
    raw = cv2.bitwise_and(raw, inner_safe)

    # Morfoloģija: aizver spraugas, atmet sīkus trokšņus
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    # Aizpildām caurumus katrā kontūrā (lai nav plankumi)
    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:  # atmetam mikropunktus
            cv2.drawContours(filled, [c], -1, 255, thickness=-1)

    teeth_mask = _smooth_mask(filled, 15)

    # --- Kvalitātes pārbaude: ja par maz/fragmentēts → fallback ---
    mouth_area = mouth_mask.sum() / 255.0
    teeth_area = teeth_mask.sum() / 255.0
    coverage = teeth_area / max(mouth_area, 1.0)

    if coverage < 0.28:
        # fallback: mutes maska bez sarkanā (lūpas/smaganas)
        red_u8 = (red_like.astype(np.uint8) * 255)
        red_u8 = cv2.dilate(red_u8, np.ones((3,3), np.uint8), 1)
        base = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(red_u8))
        base = cv2.erode(base, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
        teeth_mask = _smooth_mask(base, 15)

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
        # Pēdējais glābiņš – balinām konservatīvi mutes zonu bez sarkanā
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        red_like = (((H <= 12) | (H >= 170)) & (S > 30)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(red_like))
        mask = _smooth_mask(mask, 15)
        m = mask > 0
        if np.any(m):
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
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
