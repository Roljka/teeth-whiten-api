# teeth_api.py
import io, os
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# ---- Parametri ----
TARGET_L = 240.0     # mērķa gaišums zobiem (0..255, 240 dod baltu bez “izdegšanas”)
ALPHA_MAX = 0.85     # cik stipri maisām zobu zonā (1.0 = tikai balinātais)
FEATHER_PX = 8       # maskas mīkstināšana pikseļos
KEEP_K = 2           # saglabāt lielākos komponentus (augša/apakša)
MIN_AREA = 80        # minimālais laukums pikseļos vienai zobu komponentei

# FaceMesh indeksu saraksti
OUTER_IDX = [61,146,91,181,84,17,314,405,321,375,291,308]
INNER_IDX = [78,95,88,178,87,14,317,402,318,324,308,415]

_mp = None
_facemesh = None

def get_facemesh():
    global _mp, _facemesh
    if _facemesh is None:
        import mediapipe as mp
        _mp = mp
        _facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return _mp, _facemesh

def landmarks_poly(lms, idxs, w, h):
    return np.array([(int(lms[i].x*w), int(lms[i].y*h)) for i in idxs], dtype=np.int32)

def poly_mask(shape_hw, poly):
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [poly], 255)
    return m

def keep_biggest(mask, k=KEEP_K, min_area=MIN_AREA):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    # (area, idx) no lielākā uz mazāko
    pairs = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num)], reverse=True)
    out = np.zeros_like(mask)
    taken = 0
    for area, idx in pairs:
        if area >= min_area:
            out[lab == idx] = 255
            taken += 1
        if taken >= k:
            break
    return out

def feather(mask, r=FEATHER_PX):
    if r <= 0:
        return (mask > 0).astype(np.float32)
    soft = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (0,0), r)
    return np.clip(soft, 0, 1)

def suppress_lip_edge(inner_mask, poly, frac=0.25):
    """Samazina masku pie augšējās lūpas malas (top 25% no mutes augstuma)."""
    ys = poly[:,1]
    top, bot = ys.min(), ys.max()
    cut = int(top + (bot - top) * frac)
    sup = inner_mask.copy().astype(np.float32)
    sup[:cut, :] *= 0.2   # stipri vājinām pie augšlūpas
    return np.clip(sup, 0, 255).astype(np.uint8)

def teeth_mask(img_bgr, inner_mask, inner_poly):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    roi = inner_mask > 0
    if roi.sum() < 50:
        return np.zeros_like(L, dtype=np.float32)

    Lr, Ar, Br, Sr = L[roi].astype(np.float32), A[roi].astype(np.float32), B[roi].astype(np.float32), S[roi].astype(np.float32)
    # Sākuma sliekšņi
    pL1, pL2 = np.percentile(Lr, 70), np.percentile(Lr, 55)
    pS1, pS2 = np.percentile(Sr, 60), np.percentile(Sr, 72)
    a_med, b_med = np.median(Ar), np.median(Br)

    def build(Lmin, Smax, a_off, b_off):
        m = (
            (L.astype(np.float32) >= Lmin) &
            (S.astype(np.float32) <= Smax) &
            (A.astype(np.float32) <= a_med + a_off) &
            (B.astype(np.float32) <= b_med + b_off) &
            roi
        )
        return (m.astype(np.uint8)*255)

    m = build(pL1, pS1, 6.0, 10.0)
    if m.sum() < 0.02*roi.sum():
        m = build(pL2, pS2, 9.0, 15.0)

    # Ja vēl par maz – KMeans iekš ROI (3 klasteri)
    if m.sum() < 120:
        coords = np.column_stack(np.where(roi))
        vals = np.column_stack([L[roi].ravel(), A[roi].ravel(), B[roi].ravel(), S[roi].ravel()]).astype(np.float32)
        K = 3 if len(vals) >= 150 else 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv2.kmeans(vals, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        # Izvēlamies klasi ar lielāko L un zemu S
        best = None; best_score = -1e9
        for i in range(K):
            Lc, Ac, Bc, Sc = centers[i]
            score = Lc - 0.8*Sc - 0.1*max(0, Ac - (a_med+6)) - 0.1*max(0, Bc - (b_med+10))
            if score > best_score:
                best_score, best = score, i
        sel = (labels.ravel() == best)
        m = np.zeros_like(inner_mask, dtype=np.uint8)
        yy, xx = coords[sel][:,0], coords[sel][:,1]
        m[yy, xx] = 255

    # Morfoloģija + komponentes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)
    m = keep_biggest(m, k=KEEP_K, min_area=max(MIN_AREA, int(roi.sum()*0.002)))

    # Lūpu malu vājināšana
    m = suppress_lip_edge(m, inner_poly, frac=0.25)

    return feather(m, FEATHER_PX)

def apply_whitening(img_bgr, soft_mask):
    if soft_mask.max() <= 0:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32)
    # delta uz mērķi
    delta = np.clip(TARGET_L - Lf, 0, 255)
    # lokāls spēks
    alpha = np.clip(soft_mask * ALPHA_MAX, 0, 1).astype(np.float32)
    L_new = np.clip(Lf + delta * alpha, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L_new, A, B]), cv2.COLOR_LAB2BGR)
    # tikai zobu zonā; ārpus – oriģināls
    alpha3 = np.dstack([alpha, alpha, alpha])
    return (alpha3*out + (1.0-alpha3)*img_bgr).astype(np.uint8)

# ---- Flask ----
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "AI Teeth Whitening – teeth-only vFinal"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    data = request.files["file"].read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Bad image"}), 400

    mp, fm = get_facemesh()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")

    h, w = img.shape[:2]
    lms = res.multi_face_landmarks[0].landmark
    inner_poly = landmarks_poly(lms, INNER_IDX, w, h)
    inner_mask = poly_mask((h, w), inner_poly)

    soft = teeth_mask(img, inner_mask, inner_poly)  # 0..1
    out = apply_whitening(img, soft)

    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg", download_name="whitened.jpg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
