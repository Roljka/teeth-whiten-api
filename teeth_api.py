import io
import math
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# OpenCV HEADLESS (nav libGL)
import cv2

# MediaPipe Face Mesh (klasiskā API – 468 punkti)
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__)
CORS(app)

# ====== MediaPipe FaceMesh startēšana vienreiz ======
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,         # dod papildu lūpu/acu detālas malas
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ====== Lūpu indeksi (ārējā + iekšējā mala) – MediaPipe 468 koordināšu shēma ======
# Avots: MP FaceMesh topology; izmantojam stabilu komplektu ap muti.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
              14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
              312, 13, 82, 81, 80, 191, 78, 95]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 409, 270, 269, 267, 0, 37, 39,
              40, 185, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405,
              314, 17, 84, 181, 91, 146, 61]

def landmarks_to_np(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def build_mouth_roi(image_bgr):
    """Atgriež (success, mouth_mask_binary, outer_poly, inner_poly)."""
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    res = face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return False, None, None, None

    face_landmarks = res.multi_face_landmarks[0].landmark
    outer = landmarks_to_np([face_landmarks[i] for i in LIPS_OUTER], w, h)
    inner = landmarks_to_np([face_landmarks[i] for i in LIPS_INNER], w, h)

    # Konstruē “mutes laukumu” = ārējais poligons mīnuss iekšējais (mutas atvērums)
    mouth = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mouth, cv2.convexHull(outer), 255)
    cv2.fillConvexPoly(mouth, cv2.convexHull(inner), 0)  # atņemam iekšējo (lūpas)

    # Neliels paplašinājums, lai aizsniegtu tālākos zobus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mouth = cv2.dilate(mouth, kernel, iterations=1)

    return True, mouth, outer, inner

def teeth_mask(image_bgr, mouth_mask):
    """Zobu binārā maska tikai mutes reģionā: Lab∩HSV + morfoloģija + komponentes."""
    # Krāstelpas
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    # Heiristiski, bet robusti sliekšņi (pielāgoti “sliktai gaismai”)
    # - zobi ir gaišāki (L augsts), mazāk dzelteni (B zemāks), mazsāturēti (S zems), pietiekami gaiši (V augsts)
    # Sliekšņus var viegli pacelt/nolaist pēc vajadzības
    cond_lab = (L >= 165) & (B <= 155) & (A >= 110) & (A <= 150)
    cond_hsv = (S <= 80) & (V >= 120)

    mask = (cond_lab & cond_hsv).astype(np.uint8) * 255
    # Iekš mutes vien
    mask = cv2.bitwise_and(mask, mask, mask=mouth_mask)

    # Morfoloģija – izlīdzinām plankumus, aizveram spraugas starp zobiem
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # Izvēlamies lielākās komponentes mutes iekšienē (zobu bloki)
    num, lbls, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        # Paturam top 4 pēc laukuma (izlaižam label 0 = fons)
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        areas.sort(key=lambda x: x[1], reverse=True)
        keep = {i for i, _ in areas[:4]}
        mask2 = np.zeros_like(mask)
        for i in keep:
            mask2[lbls == i] = 255
        mask = mask2

    # Feather – lai malas nav “uzkrāsotas”
    mask_blur = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask_blur

def apply_whitening(image_bgr, mask_feather, level):
    """Balināšana Lab telpā (L↑, B↓) tikai maskā. level 1..8."""
    level = int(max(1, min(8, level)))
    # Kartējam līmeni uz L un B korekciju
    # maigāki zem 4, agresīvāki 5–8
    delta_L = [6, 10, 14, 18, 22, 26, 30, 34][level-1]
    delta_B = [6, 10, 14, 18, 22, 26, 30, 34][level-1]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Feather normēšana 0..1
    alpha = (mask_feather.astype(np.float32) / 255.0).clip(0, 1)

    # Palielinām L, samazinām B
    Lf = L.astype(np.float32) + alpha * delta_L
    Bf = B.astype(np.float32) - alpha * delta_B

    Lf = np.clip(Lf, 0, 255).astype(np.uint8)
    Bf = np.clip(Bf, 0, 255).astype(np.uint8)

    lab_out = cv2.merge([Lf, A, Bf])
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return bgr_out

def read_image_from_request():
    """Pieņem multipart form-data: lauks 'file' (vai 'image' saderībai)."""
    f = request.files.get('file') or request.files.get('image')
    if not f:
        return None, jsonify({"error": "missing file field 'file' (multipart/form-data)"}), 400
    data = f.read()
    pil = Image.open(io.BytesIO(data)).convert('RGB')
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr, None, None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/whiten", methods=["POST"])
def whiten():
    # Līmenis 1..8 (default 6)
    try:
        level = int(request.args.get("level", "6"))
    except:
        level = 6

    debug = request.args.get("debug", "0") == "1"

    image_bgr, err, code = read_image_from_request()
    if err is not None:
        return err, code

    ok, mouth_mask, outer, inner = build_mouth_roi(image_bgr)
    if not ok:
        return jsonify({"error": "mouth_not_found"}), 400

    mask = teeth_mask(image_bgr, mouth_mask)

    if debug:
        # Vizualizējam – tikai diagnostikai
        vis = image_bgr.copy()
        overlay = np.zeros_like(vis)
        overlay[:, :] = (255, 0, 255)  # rozā
        a = (mask.astype(np.float32) / 255.0 * 0.35)[..., None]
        vis = (vis * (1 - a) + overlay * a).astype(np.uint8)

        # Uzzīmējam lūpu poligonu kontūras
        cv2.polylines(vis, [cv2.convexHull(outer)], True, (0, 255, 0), 2)
        cv2.polylines(vis, [cv2.convexHull(inner)], True, (0, 200, 255), 2)

        out_img = vis
    else:
        out_img = apply_whitening(image_bgr, mask, level)

    # Atgriežam JPEG
    out_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(out_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92, subsampling=0)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
