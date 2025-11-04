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

# --- MediaPipe setup (viens instants procesam) ---
mp_face_mesh = mp.solutions.face_mesh
# Static image mode, max 1 face, augsta kvalitāte
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,     # precīzāki lūpu zīmējumi
    min_detection_confidence=0.6
)

# Iekšējās lūpas (mouth inner) landmarķi FaceMesh 468 sistēmā
# (klasiskā indeksa kopa ap iekšējo lūpu atvērumu)
INNER_LIP_IDX = np.array([
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,
    87,178,88,95
], dtype=np.int32)

def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask, k=11):
    if k % 2 == 0: k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask

def _build_mouth_mask(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]
    inner = _landmarks_to_xy(landmarks, w, h, INNER_LIP_IDX)

    area = cv2.contourArea(inner)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner], 255)

    # MAZĀKA paplašināšana un bez agresīvas pabīdīšanas uz leju
    dil = max(8, int(math.sqrt(area) * 0.03))        # bija 0.04
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # vairs NEBīdam vertikāli (neradām lūpu paķeršanu)
    mask = _smooth_mask(mask, 17)
    return mask


def _build_teeth_mask(img_bgr, mouth_mask):
    """Cieša zobu maska (bez lūpām/smaganām)."""
    if mouth_mask.sum() == 0:
        return mouth_mask

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    m = mouth_mask > 0
    Lm = L[m]; Am = A[m]; Bm = B[m]
    if Lm.size == 0:
        return np.zeros_like(mouth_mask)

    # Relatīvi sliekšņi pret mutes zonu
    L_thr = np.percentile(Lm, 60)        # zobi parasti gaišāki
    B_thr = np.percentile(Bm, 55)        # zemāks B = mazāk dzeltena
    A_max = 150                          # rozā/sarkans (smaganas) virs šī

    teeth0 = ((L > L_thr) & (B < B_thr) & (A < A_max) & m).astype(np.uint8) * 255

    # Noņemam mutes iekšējās robežas joslu (edge-guard), lai netrāpītu lūpām
    EDGE_GUARD = 4
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_GUARD*2+1, EDGE_GUARD*2+1))
    inner_shrunk = cv2.erode(mouth_mask, k, iterations=1)   # mutes maska, sašaurināta no malām
    teeth1 = cv2.bitwise_and(teeth0, inner_shrunk)

    # Mazie trokšņi prom + padarām vienmērīgāku
    teeth1 = cv2.morphologyEx(teeth1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    # Piespiežam masku pie zobiem (mazliet saraujam prom no lūpām/smaganām)
    ERODE_PX = 2
    if ERODE_PX > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX*2+1, ERODE_PX*2+1))
        teeth1 = cv2.erode(teeth1, k2, iterations=1)

    # Neliels close, lai aizvērtu spraudziņas starp zobiem (ne pārāk!)
    teeth1 = cv2.morphologyEx(teeth1, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    return _smooth_mask(teeth1, 15)


def _teeth_whiten(img_bgr):
    h, w = img_bgr.shape[:2]

    # MediaPipe detekcija (RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)

    if not res.multi_face_landmarks:
        # nav sejas — atgriež oriģinālu
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark

    # 1) Mutes iekšējā maska
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)

    # 2) cieša zobu maska no krāsām mutes iekšienē
teeth_mask = _build_teeth_mask(img_bgr, mouth_mask)

if np.sum(teeth_mask) == 0:
    return img_bgr
    
    # 3) Rezerves “invert trick” maska — paņemam tumšākās zonas inversā
    inv = 255 - img_bgr
    inv_hsv = cv2.cvtColor(inv, cv2.COLOR_BGR2HSV)
    # tumši-zilgani (zobi oriģinālā gaiši) ⇒ inv būs ar H~zils un V zemāks; te mīksts logs
    lower = np.array([80, 20, 20], dtype=np.uint8)
    upper = np.array([140, 255, 180], dtype=np.uint8)
    inv_mask = cv2.inRange(inv_hsv, lower, upper)

    # 3) Apvienojam: tikai tur, kur ir mute, un kur inv and/or mute sniedz signālu
    raw_mask = cv2.bitwise_and(mouth_mask, inv_mask)
    # ja inv_mask pārāk vāja, izmanto tikai mouth_mask
    mask_sum = np.sum(raw_mask) / 255
    if mask_sum < 800:  # drošs slieksnis
        raw_mask = mouth_mask.copy()

    # 4) Izmetam smaganas ar LAB ‘a’ kanālu (rozā/sarkans)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # smaganas mēdz būt A > ~145 (jo 128~neitrāls; > augstāks siltums/rozā)
    gums = (A > 145).astype(np.uint8) * 255
    gums = cv2.dilate(gums, np.ones((3,3), np.uint8), iterations=1)
    teeth_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(gums))
    teeth_mask = _smooth_mask(teeth_mask, 31)

    if np.sum(teeth_mask) == 0:
        return img_bgr

    # 5) Balināšana: palielinam L (gaišums) + mazinām B (dzeltenums)
    # Konservatīvi, bet pamanāmi; vēlāk vari pielāgot konstantes
    L_f = L.astype(np.float32)
    B_f = B.astype(np.float32)

    # tikai tur, kur maska
    m = (teeth_mask > 0)

    # gaišums (gamma + gain)
    L_f[m] = np.clip(L_f[m] * 1.15 + 12, 0, 255)

    # dzeltenuma samazināšana
    B_f[m] = np.clip(B_f[m] * 0.82 - 8, 0, 255)

    # nedaudz trokšņu izlīdzināšanas zobu iekšienē
    L_out = L_f.astype(np.uint8)
    B_out = B_f.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B_out])

    # tikai mutes zonā liksim atpakaļ, citur — oriģinālais LAB
    lab_final = lab.copy()
    lab_final[m] = lab_out[m]

    out_bgr = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    # Neliels bilaterāls izlīdzinājums tikai maskas robežās (novāc plankumus)
    blur = cv2.bilateralFilter(out_bgr, d=7, sigmaColor=40, sigmaSpace=40)
    out_bgr[m] = blur[m]

    return out_bgr

def _read_image_from_request():
    # pieņem form-data 'file' VAI 'image'
    if 'file' in request.files:
        f = request.files['file']
    elif 'image' in request.files:
        f = request.files['image']
    else:
        return None, ("missing file field 'file' (multipart/form-data)", 400)

    # saglabā EXIF orientāciju
    try:
        pil = Image.open(f.stream)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        return None, (f"cannot open image: {e}", 400)

    img = np.array(pil)[:, :, ::-1]  # RGB->BGR
    return img, None

@app.route("/whiten", methods=["POST"])
def whiten():
    img_bgr, err = _read_image_from_request()
    if err:
        return jsonify({"error": err[0]}), err[1]

    out_bgr = _teeth_whiten(img_bgr)

    # atpakaļ uz RGB un saspiežam kā JPEG
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
