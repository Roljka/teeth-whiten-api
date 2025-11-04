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

# ---- TUNING PARAMS ----
DIL_H_SCALE = 0.055   # horizontālā dilatācija (vairāk, lai aizsniegtu sānu zobus)
DIL_V_SCALE = 0.020   # vertikālā dilatācija (mazāk, lai neuzkāptu uz lūpām)
SHIFT_DOWN   = 0      # px pabīde lejup no iekš-lūpas (0…2). 0 = turam cieši.
GUMS_A_MIN   = 150    # LAB A kanāls; > šī vērtība tipiski ir smaganas (rozā/sarkans)
TEETH_L_MIN  = 55     # minimālais gaišums zobiem mutes zonā
TEETH_B_MAX  = 165    # maksimālais “dzeltenums” zobiem
KEEP_CC      = 4      # cik lielos savienotos reģionus paturam
FEATHER_PX   = 7      # maskas “spalva” (robežas mīkstināšanai)
CLAHE_CLIP   = 2.0    # CLAHE kontrasta klips (zobu normalizēšanai)
BALANCE_A    = 0.15   # A kanāla korekcijas intensitāte (aizvelkam no rozā uz neitrālu)
YELLOW_REDU  = 0.88   # B kanāla dzeltenuma mazinājums (0.8–0.92)
L_GAIN       = 0.55   # cik stipri celt L (0.4–0.7) – “dabiskumam”
L_TARGET     = 235    # pie kā mērķojam augšējo kvartili (230–240)

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

    # Bāzes maska – iekšējā lūpa
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    # Anizotropa dilatācija: horizontāli vairāk, vertikāli mazāk
    kx = max(5, int(math.sqrt(area) * DIL_H_SCALE))
    ky = max(3, int(math.sqrt(area) * DIL_V_SCALE))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx | 1, ky | 1))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # (neobligāti) neliels pabīdiens uz leju, ja vajag “ielīst zem ēnām”
    if SHIFT_DOWN > 0:
        M = np.float32([[1, 0, 0], [0, 1, SHIFT_DOWN]])
        mask = cv2.warpAffine(mask, M, (w, h))

    # Izslēdzam acīmredzami rozā (smaganas) ar LAB A kanālu
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)
    gums = (A > GUMS_A_MIN).astype(np.uint8) * 255
    gums = cv2.dilate(gums, np.ones((3, 3), np.uint8), 1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(gums))

    # Nedaudz sašaujam vertikāli (lai nelīstu uz lūpām)
    if ky > 1:
        trim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, trim, 1)

    # Feather
    mask = cv2.GaussianBlur(mask, (FEATHER_PX | 1, FEATHER_PX | 1), 0)
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
    # cv2.threshold prasa 2D/attēlu – uztaisām mazu karti
    L_thr_img = L_roi.reshape(-1, 1).astype(np.uint8)     # FIX: no 1D -> 2D
    B_thr_img = B_roi.reshape(-1, 1).astype(np.uint8)     # FIX
    _, L_thr = cv2.threshold(L_thr_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, B_thr = cv2.threshold(B_thr_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    L_thr = int(L_thr)  # skaitlis
    B_thr = int(B_thr)

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
    cnts = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(filled, [c], -1, 255, -1)

    teeth_mask = _smooth_mask(filled, 13)

    # kvalitātes kontrole – ja par maz pārklājuma, drošais fallback (mute bez smaganām)
    mouth_area = mouth_mask.sum() / 255.0
    teeth_area = teeth_mask.sum() / 255.0
    ratio = teeth_area / max(mouth_area, 1.0)            # FIX: aizsardzība
    if ratio < 0.30:
        base = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(gums.astype(np.uint8)*255))
        base = cv2.erode(base, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
        teeth_mask = _smooth_mask(base, 13)

    return teeth_mask

def _teeth_whiten(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark

    # 1) Plaša mutes maska, kas respektē smaganas
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    if np.sum(mouth_mask) == 0:
        return img_bgr

    # 2) Zobu kandidāti pēc LAB (ne “krāsojam”, bet atlasām)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    teeth_cand = (L > TEETH_L_MIN) & (B < TEETH_B_MAX) & (A < GUMS_A_MIN)
    teeth_cand = teeth_cand.astype(np.uint8) * 255
    teeth_cand = cv2.bitwise_and(teeth_cand, mouth_mask)

    # Stabilizācija – ņemam tikai lielākos savienotos reģionus
    num, labels, stats, _ = cv2.connectedComponentsWithStats(teeth_cand, 8)
    keep = np.zeros_like(teeth_cand)
    if num > 1:
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        areas.sort(key=lambda x: x[1], reverse=True)
        for i, _a in areas[:KEEP_CC]:
            keep[labels == i] = 255
    teeth_mask = keep

    if np.sum(teeth_mask) == 0:
        # rezerves – ja ļoti tumšs: ņemam vismaz mutes centru
        teeth_mask = mouth_mask.copy()

    # Feather/clean
    teeth_mask = cv2.morphologyEx(
        teeth_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1
    )
    teeth_mask = cv2.GaussianBlur(teeth_mask, (FEATHER_PX | 1, FEATHER_PX | 1), 0)
    m = teeth_mask > 0

    # 3) Normalizējam apgaismojumu iekš maskas (mazāk plankumu sliktā gaismā)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    # FIX: CLAHE jālieto uz 2D attēla. Uztaisām pilnam L un pēc tam ieliekam tikai maskā.
    L_eq_full = clahe.apply(L)         # 2D
    L_eq = L.copy()
    L_eq[m] = L_eq_full[m]

    # 4) Dabīgs “balinājums”: paceļam L uz mērķi, mazinām B, A tuvojam neitralitātei
    if np.any(m):
        p95 = np.percentile(L_eq[m], 95)
    else:
        p95 = 200
    if p95 < 1:
        p95 = 1
    gain = L_GAIN * (L_TARGET / float(p95))
    L_new = L_eq.astype(np.float32)
    L_new[m] = np.clip(L_eq[m] * gain, 0, 255)

    # dzeltenuma mazinājums
    B_new = B.astype(np.float32)
    B_new[m] = np.clip(B[m] * YELLOW_REDU - 4, 0, 255)

    # A kanalā nedaudz velkam uz 128 (mazāk rozā)
    A_new = A.astype(np.float32)
    A_new[m] = np.clip(128 + (A[m] - 128) * (1.0 - BALANCE_A), 0, 255)

    # 5) Rekonstrukcija un faktūras saglabāšana (maigs blend)
    lab_new = cv2.merge([L_new.astype(np.uint8),
                         A_new.astype(np.uint8),
                         B_new.astype(np.uint8)])
    bgr_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    # neliels edge-preserving blur tikai zobu zonā, pēc tam sajaukšana ar oriģinālu
    smooth = cv2.bilateralFilter(bgr_new, d=7, sigmaColor=40, sigmaSpace=40)
    out = img_bgr.copy()
    alpha = (teeth_mask.astype(np.float32) / 255.0)[..., None]  # [H,W,1]
    out = (out * (1 - alpha) + smooth * alpha).astype(np.uint8)
    return out

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
