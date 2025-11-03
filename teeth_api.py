import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ---------------- Flask ----------------
app = Flask(__name__)
CORS(app)

# --------------- Mediapipe (lazy import) ---------------
_mp = None
_face_mesh = None

def ensure_mp():
    global _mp, _face_mesh
    if _mp is None or _face_mesh is None:
        import mediapipe as mp
        _mp = mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )

# ----------------- Utils -----------------
def pil_to_bgr(pil_img):
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_to_bgr(img)

def lips_mask_from_landmarks(h, w, landmarks):
    idx = set()
    for a, b in _mp.solutions.face_mesh.FACEMESH_LIPS:
        idx.add(a); idx.add(b)
    pts = []
    for i in idx:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask

def shrink(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.erode(mask, k, iterations=1)

def grow(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.dilate(mask, k, iterations=1)

# --------- Teeth mask pipeline (stable+dark-light add-ons) ---------
def build_teeth_mask(bgr, lips_mask):
    h, w = bgr.shape[:2]

    # Mutes iekšpuse – atvirzāmies no lūpām (lai neskaram lūpu kontūru)
    mouth_inner = shrink(lips_mask, px=max(1, min(h, w)//300))  # 1-3px tipiski

    # Vadības attēli (tikai maskas noteikšanai)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # CLAHE tikai L kanālam iekš mutes – palīdz ēnās
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_eq = L.copy()
    L_eq[mouth_inner > 0] = clahe.apply(L[mouth_inner > 0])

    # Dinamiskie sliekšņi iekš mutes
    mouth_idxs = mouth_inner > 0
    if np.count_nonzero(mouth_idxs) < 50:
        return np.zeros((h, w), np.uint8)

    S_m = S[mouth_idxs].astype(np.float32)
    V_m = V[mouth_idxs].astype(np.float32)
    # ņemam zemo S un augsto V percentiļus (robustrāk pret tumšām zonām)
    s_thr = float(np.percentile(S_m, 55))   # nedaudz virs mediānas
    v_thr = float(np.percentile(V_m, 45))   # nedaudz zem mediānas

    base = (S <= s_thr) & (V >= v_thr) & mouth_idxs

    # Izgriežam smaganas/lūpas: sarkanā josla + augsts A (rozā/gaļīgais)
    red_like = (((H <= 12) | (H >= 170)) & (S > 30))
    gum_like = (A >= 155)  # rozā tonis
    base = base & (~red_like) & (~gum_like)

    # Ēnu atguve: adaptīvs slieksnis uz L_eq iekš mutes + savienošana ar base
    L_local = np.zeros_like(L)
    L_local[mouth_idxs] = L_eq[mouth_idxs]
    # adaptīvs – paņem gaišākās emaljas zonas arī ēnās
    adap = cv2.adaptiveThreshold(
        L_local, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        35, -5
    )
    adap = (adap > 0) & mouth_idxs
    cand = base | adap

    # Morfoloģija, lai noņem troksni un aizver spraugas
    mask = np.zeros((h, w), np.uint8)
    mask[cand] = 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.bitwise_and(mask, mouth_inner)  # drošībai

    # Sēklu ģenerēšana (centroidi) un region-grow uz L_eq,
    # lai “pielipinātu” blakus pikseļus ar līdzīgu gaišumu
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    seeds = []
    for i in range(1, num):
        x, y = int(centroids[i][0]), int(centroids[i][1])
        if 0 <= x < w and 0 <= y < h and mouth_inner[y, x] > 0:
            seeds.append((x, y))
    if not seeds:
        # ja sākumā nav komponentu, sēklu vietā ņem mutes apakš-vidus daļu
        ys, xs = np.where(mouth_inner > 0)
        if len(xs) > 0:
            mid = np.argmin(np.abs(ys - np.median(ys)))
            seeds = [(int(xs[mid]), int(ys[mid]))]

    grow_mask = np.zeros((h+2, w+2), np.uint8)  # floodfill maskai jābūt +2
    grown = np.zeros((h, w), np.uint8)
    # tolerances ēnām (jo tumšāks, jo lielāku T dodam)
    l_std = float(np.std(L_eq[mouth_idxs])) + 1.0
    tol = max(6, min(18, int(l_std)))  # 6..18 diapazons

    L_ff = L_eq.copy()
    for (sx, sy) in seeds[:4]:  # pietiek ar pāris sēklām
        grow_mask[:] = 0
        # floodfill tikai mutes kontūrā, izmantojot masku un toleranci
        cv2.floodFill(L_ff, grow_mask, (sx, sy), 255,
                      loDiff=tol, upDiff=tol,
                      flags=(4 | (255 << 8)))
        # floodfill maskā aizpildītā zona = 1 iekš (grow_mask > 0)
        grown |= (grow_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255

    # Savienojam sākotnējo masku ar region-grow rezultātu, bet tikai mutes iekšpusē
    mask = cv2.bitwise_and(grown, mouth_inner)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    # Atstājam 2 lielākos laukumus (augša/apakša)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = []
        for i in range(1, num_labels):
            areas.append((int(stats[i, cv2.CC_STAT_AREA]), i))
        areas.sort(reverse=True)
        keep = [idx for (_, idx) in areas[:2]]
        filt = np.zeros_like(mask)
        for i in keep:
            filt[labels == i] = 255
        mask = filt

    # Droši nebalinām smaganas: izmetam pikseļus ar augstu A vai sarkanu H
    gum_cut = ((A >= 155) | (((H <= 12) | (H >= 170)) & (S > 30)))
    mask[gum_cut] = 0

    # Nofeiderojam malas (gludas pārejas)
    blurred = cv2.GaussianBlur(mask, (0,0), sigmaX=1.2, sigmaY=1.2)
    blurred = (blurred / 255.0).clip(0, 1)
    return (blurred * 255).astype(np.uint8)

def whiten_Lab(bgr, soft_mask, l_gain=14, b_shift=22):
    if soft_mask is None or soft_mask.ndim != 2 or np.max(soft_mask) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    alpha = (soft_mask.astype(np.float32) / 255.0)  # 0..1

    # tikai tajās zonās, kur alpha > 0
    Lf = L.astype(np.float32)
    Bf = B.astype(np.float32)
    Lf = np.clip(Lf + l_gain * alpha, 0, 255)
    Bf = np.clip(Bf - b_shift * alpha, 0, 255)

    out = cv2.cvtColor(cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)]), cv2.COLOR_LAB2BGR)
    return out

# ------------------- Endpoints -------------------
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data with field 'file'."), 400

        ensure_mp()

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        # Face mesh
        res = _face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        lips_mask = lips_mask_from_landmarks(h, w, landmarks)

        soft_mask = build_teeth_mask(bgr, lips_mask)
        out = whiten_Lab(bgr, soft_mask, l_gain=14, b_shift=22)

        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify(error="Encode failed"), 500

        return send_file(io.BytesIO(buf.tobytes()),
                         mimetype="image/jpeg",
                         as_attachment=False,
                         download_name="whitened.jpg")
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
