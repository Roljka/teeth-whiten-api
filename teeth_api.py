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

# ---------- Tuning parametri (brīvi grozāmi) ----------
DIL_H_SCALE = 0.080   # horizontālā paplašināšana (vairāk -> ķer sānu zobus)
DIL_V_SCALE = 0.025   # vertikālā paplašināšana (mazāk -> netrāpa lūpās)
EDGE_GUARD   = 4      # cik stipri “ierobežot” malu pret lūpām (px mērogots ar elipsi)
FEATHER_PX   = 15     # maskas mīkstināšana (gauss)
A_MAX        = 148    # LAB A kanāla griesti (virs tā uzskatām par rozā/sarkanu)
RED_H_LOW    = 12     # HSV: sarkanais zem 12°
RED_H_HIGH   = 170    # HSV: sarkanais virs 170°
RED_S_MIN    = 28     # HSV S piesātinājuma minimums, lai skaitītu kā lūpu/smaganu sarkano
L_DELTA      = -15    # pazeminām Otsu L slieksni (ļauj tumšākus zobus)
B_DELTA      = +22    # paaugstinām Otsu B slieksni (ļauj dzeltenākus zobus)
MIN_TOOTH_CC = 80     # min kontūra laukums (px) pēc maskas

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

    # bāzes maska no iekšējās lūpas
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    # anizotropa dilatācija — plašāk horizontāli, mazāk vertikāli
    side = math.sqrt(area)
    kx = max(9, int(side * DIL_H_SCALE)) | 1
    ky = max(5, int(side * DIL_V_SCALE)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # malas aizsardzība (neuzkāpt uz lūpām)
    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_safe = cv2.erode(mask, guard_k, iterations=1)

    mask = _smooth_mask(inner_safe, FEATHER_PX)
    return mask

def _build_teeth_mask(img_bgr, mouth_mask):
    """
    Zobu maska mutes iekšienē ar adaptīviem sliekšņiem un sarkanā (lūpas/smaganas) izslēgšanu.
    """
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, _V = cv2.split(hsv)

    m = mouth_mask > 0
    if not np.any(m):
        return np.zeros_like(mouth_mask)

    # Otsu sliekšņi tikai mutes zonai
    L_roi = L[m]; B_roi = B[m]
    L_thr, _ = cv2.threshold(L_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    B_thr, _ = cv2.threshold(B_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Pielaidīgāki zobiem tumšā/ dzeltenā apgaismojumā
    L_thr = max(40, int(L_thr) + L_DELTA)   # mazāks slieksnis -> iekļaujam tumšākus
    B_thr = min(220, int(B_thr) + B_DELTA)  # lielāks slieksnis -> iekļaujam dzeltenākus

    # Sarkanais (lūpas/smaganas) no HSV + LAB
    red_hsv = (((H <= RED_H_LOW) | (H >= RED_H_HIGH)) & (S >= RED_S_MIN))
    red_lab = (A >= A_MAX)
    red_like = (red_hsv | red_lab)

    # Kandidāti: gaiši, nedzelteni, ne sarkani, un mutes zonā
    raw = ((L > L_thr) & (B < B_thr) & (~red_like) & m).astype(np.uint8) * 255

    # Edge guard: iekšējā drošības josla pret lūpām
    guard_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_safe = cv2.erode(mouth_mask, guard_k, iterations=1)
    raw = cv2.bitwise_and(raw, inner_safe)

    # Morfoloģija un caurumu aizpilde
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > MIN_TOOTH_CC:
            cv2.drawContours(filled, [c], -1, 255, thickness=-1)

    teeth_mask = _smooth_mask(filled, 13)

    # Ja pārklājums pārāk mazs — fallback: mutes maska bez “sarkanā”
    mouth_area = mouth_mask.sum() / 255.0
    teeth_area = teeth_mask.sum() / 255.0
    if teeth_area / max(mouth_area, 1.0) < 0.28:
        red_u8 = (red_like.astype(np.uint8) * 255)
        red_u8 = cv2.dilate(red_u8, np.ones((3,3), np.uint8), 1)
        base = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(red_u8))
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
        # pēdējais glābiņš – balinām konservatīvi mutes zonu bez “sarkanā”
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, _V = cv2.split(hsv)
        red_like = (((H <= RED_H_LOW) | (H >= RED_H_HIGH)) & (S >= RED_S_MIN)) | (A >= A_MAX)
        mask = cv2.bitwise_and(mouth_mask, cv2.bitwise_not((red_like.astype(np.uint8) * 255)))
        mask = _smooth_mask(mask, FEATHER_PX)
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

    # Balināšana (dabīgs looks)
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
