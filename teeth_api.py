import io
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from PIL import Image

# ========= Helpers =========

def shrink(mask: np.ndarray, px: int = 1) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
    return cv2.erode(mask, k, iterations=1)

def polygon_mask(shape, pts):
    mask = np.zeros(shape[:2], np.uint8)
    if pts is None or len(pts) < 3:
        return mask
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask

def lips_masks_from_facemesh(img_bgr):
    """ MediaPipe FaceMesh -> outer lips - inner mouth poligoni.
        Atgriež (lips_outer, lips_inner, ok)
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        res = face_mesh.process(img_rgb)

    if not res.multi_face_landmarks:
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), False

    lm = res.multi_face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float32)

    # MediaPipe indikatori (standarta komplekts)
    OUTER = np.array([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61])
    INNER = np.array([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 78])

    lips_outer = polygon_mask(img_bgr.shape, pts[OUTER])
    lips_inner = polygon_mask(img_bgr.shape, pts[INNER])

    # Mutes “iekšpuse” = ārējā lūpa MĪNUS iekšējā (zobu/iekšmutes logs)
    mouth_window = cv2.bitwise_and(lips_outer, cv2.bitwise_not(lips_inner))

    return lips_outer, mouth_window, True

def build_teeth_mask(bgr, mouth_mask):
    """ Stabilā, plankumus salīmējošā versija ar 'invert-zaļās' palīdzību. """
    h, w = bgr.shape[:2]
    if mouth_mask is None or mouth_mask.size == 0:
        return np.zeros((h, w), np.uint8)

    mouth_inner = shrink(mouth_mask, px=max(1, min(h, w)//300))
    mouth = (mouth_inner > 0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # --- invert-helper: lūpas/smaganas kļūst zaļganas, zobi paliek ne-zaļi
    inv = 255 - bgr
    inv_hsv = cv2.cvtColor(inv, cv2.COLOR_BGR2HSV)
    iH, iS, iV = cv2.split(inv_hsv)
    inv_green = ((iH >= 40) & (iH <= 100) & (iS > 60))
    non_green_mouth = (~inv_green) & mouth

    # --- CLAHE ēnām (UZ VISAS L, pēc tam pa mutes masku paņemam)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq_all = clahe.apply(L)  # <-- 2D ievade CLAHE, izvairāmies no boolean index assignment bēdām
    L_eq = np.where(mouth, L_eq_all, L)  # pa mutes masku - equalized, citur oriģināls

    # --- adaptīvie sliekšņi S/V mutes iekšienē
    if np.count_nonzero(mouth) < 50:
        return np.zeros((h, w), np.uint8)

    S_m = S[mouth].astype(np.float32)
    V_m = V[mouth].astype(np.float32)
    s_thr = float(np.percentile(S_m, 55))  # mazāka piesāt., baltāki
    v_thr = float(np.percentile(V_m, 45))  # gaišāki par šo

    base_teeth = (S <= s_thr) & (V >= v_thr) & mouth

    # --- izgriež smaganas/lūpas pec oriģ. "sarkanā" + LAB A*
    red_like = (((H <= 12) | (H >= 170)) & (S > 30))
    gum_like = (A >= 155)

    # --- adaptīvs L slieksnis (lokāls)
    L_local = np.zeros_like(L)
    L_local[mouth] = L_eq[mouth]
    adap = cv2.adaptiveThreshold(L_local, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, -5) > 0

    # Kandidāti = (S/V zobi) VAI (adap L), tikai ne-zaļajā mutes apgabalā
    cand = (base_teeth | (adap & mouth)) & non_green_mouth
    cand &= (~red_like) & (~gum_like)

    mask = np.zeros((h, w), np.uint8)
    mask[cand] = 255

    # Morfoloģiskā tīrīšana
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.bitwise_and(mask, mouth_inner)

    # --- Region-grow uz L_eq (salīmē plankumus)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    seeds = []
    for i in range(1, num):
        x, y = int(centroids[i][0]), int(centroids[i][1])
        if 0 <= x < w and 0 <= y < h and mouth_inner[y, x] > 0:
            seeds.append((x, y))
    if not seeds:
        ys, xs = np.where(mouth)
        if len(xs) > 0:
            mid = int(len(xs) / 2)
            seeds = [(int(xs[mid]), int(ys[mid]))]

    grow_mask = np.zeros((h+2, w+2), np.uint8)
    grown = np.zeros((h, w), np.uint8)
    l_std = float(np.std(L_eq[mouth])) + 1.0
    tol = max(6, min(18, int(l_std)))
    L_ff = L_eq.copy()
    for (sx, sy) in seeds[:4]:
        grow_mask[:] = 0
        cv2.floodFill(L_ff, grow_mask, (sx, sy), 255,
                      loDiff=tol, upDiff=tol, flags=(4 | (255 << 8)))
        grown |= ((grow_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255)

    mask = cv2.bitwise_and(grown, mouth_inner)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    # Atstājam 2 lielākās komponentes (augša+apakša)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num_labels)]
        areas.sort(reverse=True)
        keep = [idx for (_, idx) in areas[:2]]
        filt = np.zeros_like(mask)
        for i in keep:
            filt[labels == i] = 255
        mask = filt

    # Drošības izgriešana – smaganas/lūpas nost
    bad = ((A >= 155) | (((H <= 12) | (H >= 170)) & (S > 30)))
    mask[bad] = 0

    # Mīksta mala
    soft = cv2.GaussianBlur(mask, (0, 0), 1.2)
    return soft

def whiten_teeth(bgr, teeth_mask, strength=0.65, blue_fix=0.45):
    """ Balina tikai maskas zonu: +L (LAB), mazina B (dzelt. komponente), mīksts blend. """
    if teeth_mask is None or np.max(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mask_f = (teeth_mask.astype(np.float32) / 255.0)
    mask_f = cv2.merge([mask_f, mask_f, mask_f])

    # Palielinām gaišumu (L) un samazinām dzeltenumu (B -> 128)
    L = L.astype(np.float32)
    B = B.astype(np.float32)

    L_new = L + strength * (255.0 - L) * (teeth_mask.astype(np.float32)/255.0)
    B_new = B - blue_fix * (B - 128.0) * (teeth_mask.astype(np.float32)/255.0)

    L_new = np.clip(L_new, 0, 255).astype(np.uint8)
    B_new = np.clip(B_new, 0, 255).astype(np.uint8)

    lab_out = cv2.merge([L_new, A, B_new])
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # Vēl neliels “specular” spīdums (uz iekšas)
    shine = cv2.GaussianBlur(teeth_mask, (0, 0), 2.0).astype(np.float32) / 255.0
    out = out.astype(np.float32)
    out = out * (1.0 + 0.08 * shine[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# ========= Flask app =========
app = Flask(__name__)
# CORS – atļaujam no jebkuras izcelsmes (vari nomainīt uz savu domēnu)
from flask_cors import CORS
CORS(app,
     resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Requested-With"])

# Globāli piešujam CORS headerus (preflight atbildei un visiem citiem)
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    return resp

# Ļaujam preflight /whiten maršrutam (bez loģikas, tikai 204)
@app.route("/whiten", methods=["OPTIONS"])
def whiten_options():
    return ("", 204)

# (pēc vajadzības) lielāku upload limitu
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "teeth-whitening", "version": "med-0.6"})

@app.route("/whiten", methods=["POST"])
def whiten():
    """
    Multipart:
      - file: image (jpg/png/webp)
      - strength: float [0..1] (optional)
      - blue_fix: float [0..1] (optional)
    """
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400

    strength = float(request.form.get("strength", "0.65"))
    blue_fix = float(request.form.get("blue_fix", "0.45"))

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1) Lips + mouth window
    lips_outer, mouth_window, ok = lips_masks_from_facemesh(bgr)
    if not ok or np.max(mouth_window) == 0:
        # Ja nespēja atrast, vispār neatliek balināt – atgriežam oriģinālu
        out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    # 2) Zobu maska (ar invert-zaļo helperi un region-grow)
    teeth_mask = build_teeth_mask(bgr, mouth_window)

    # 3) Balināšana
    out_bgr = whiten_teeth(bgr, teeth_mask, strength=strength, blue_fix=blue_fix)

    out = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


if __name__ == "__main__":
    # Lokālai testēšanai
    app.run(host="0.0.0.0", port=10000, debug=False)
