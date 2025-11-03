import io, os, cv2, numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# ── I/O ───────────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img):
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return pil_to_bgr(img)

# ── Lips / mouth masks ────────────────────────────────────────────────────────
def lips_hull_mask(h, w, landmarks):
    idx = set()
    for a, b in mp_face_mesh.FACEMESH_LIPS:
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

def erode(mask, px):
    if px <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.erode(mask, k, 1)

def dilate(mask, px, shape="ellipse"):
    if px <= 0: return mask
    kshape = cv2.MORPH_ELLIPSE if shape=="ellipse" else cv2.MORPH_RECT
    k = cv2.getStructuringElement(kshape, (2*px+1, 2*px+1))
    return cv2.dilate(mask, k, 1)

def hsv_red_like(bgr, a_thresh):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, _ = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)
    red = (((H <= 12) | (H >= 170)) & (S > 30)) | (A > a_thresh)
    out = np.zeros(H.shape, np.uint8); out[red] = 255
    return out

def build_mouth_safe(bgr, landmarks, lowlight):
    h, w = bgr.shape[:2]
    lips = lips_hull_mask(h, w, landmarks)
    edge = max(1, min(h, w)//240)                 # 1–3 px
    inner = erode(lips, edge)                      # atvirzāmies no lūpu robežas
    # mazliet vertikāli paplašinām uz leju (lai paņem zobu apakšas)
    deep = cv2.dilate(inner, cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 5 if lowlight else 4)), 1)
    # izmetam rozā/sarkano (lūpas, smaganas)
    red = hsv_red_like(bgr, a_thresh=(162 if lowlight else 158))
    safe = cv2.bitwise_and(deep, cv2.bitwise_not(red))
    # izlīdzinājumi
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)), 2)
    if np.count_nonzero(safe) < 200:
        safe = inner
    return safe

# ── CLAHE tikai mutē (uzlabo ēnas, FIX dimensiju kļūdu) ─────────────────────
def clahe_in_mouth(bgr, safe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_eq = clahe.apply(L)                 # 2D kadrs
    L2 = L.copy()
    m = safe.astype(bool)
    L2[m] = L_eq[m]
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

# ── Mahalanobis klasifikācija mutē (robusta ēnām) ────────────────────────────
def classify_teeth_mahalanobis(bgr, safe, lowlight):
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]

    m = safe > 0
    if np.count_nonzero(m) < 200:
        return np.zeros((h, w), np.uint8)

    # sēklas: gaišākie “ne-rozā”
    seed = m & (A < (162 if lowlight else 158)) & (V > (80 if lowlight else 92))
    ys, xs = np.where(seed)
    if xs.size == 0:
        return np.zeros((h, w), np.uint8)
    # paņemam top-N pēc L
    idx = np.argsort(-L[ys, xs])[: (450 if lowlight else 300)]
    sy, sx = ys[idx], xs[idx]

    # statistika
    S = np.stack([L[sy,sx].astype(np.float32),
                  A[sy,sx].astype(np.float32),
                  B[sy,sx].astype(np.float32)], axis=1)
    mu = S.mean(axis=0)
    cov = np.cov(S.T) + np.eye(3, dtype=np.float32)*1.5  # regularizācija
    inv = np.linalg.inv(cov)

    # distances visiem mutē
    P = np.stack([L[m].astype(np.float32),
                  A[m].astype(np.float32),
                  B[m].astype(np.float32)], axis=1)
    d = np.sqrt(((P - mu) @ inv) * (P - mu)).sum(axis=1)

    # adaptīvs slieksnis: sēklu 85. percentile + neliels buferis
    ds = np.sqrt(((S - mu) @ inv) * (S - mu)).sum(axis=1)
    thr = np.percentile(ds, 85.0) + (0.9 if lowlight else 0.7)

    mask = np.zeros((h, w), np.uint8)
    mm = np.zeros_like(m, dtype=np.bool_)
    mm[m] = d <= thr
    mask[mm] = 255

    # nolaužam sarkanīgo (smaganas) drošībai
    gum_like = (A > (160 if lowlight else 156))
    mask[gum_like] = 0

    # “row-fill” un “column-fill”, lai nebūtu puszobi
    mask = cv2.bitwise_and(mask, safe)
    filled = np.zeros_like(mask)
    rows = np.where(np.sum(mask>0, axis=1) > 0)[0]
    for y in rows:
        xs = np.where(mask[y]>0)[0]
        if xs.size >= 2:
            filled[y, xs.min():xs.max()+1] = 255
    cols = np.where(np.sum(mask>0, axis=0) > 0)[0]
    for x in cols:
        ys = np.where(mask[:,x]>0)[0]
        if ys.size >= 2:
            filled[ys.min():ys.max()+1, x] = 255

    # distance-close + viegla vertikāla paplašināšana
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)), 2)
    vpush = dilate(filled, 1, "rect")
    filled = cv2.bitwise_and(vpush, safe)
    return cv2.medianBlur(filled, 3)

# ── Balināšana ────────────────────────────────────────────────────────────────
def whiten_lab(bgr, mask, l_gain=14, b_shift=22, a_pull=4):
    if np.count_nonzero(mask) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = mask > 0
    L = L.astype(np.int16); A = A.astype(np.int16); B = B.astype(np.int16)
    L[m] = np.clip(L[m] + l_gain, 0, 255)
    B[m] = np.clip(B[m] - b_shift, 0, 255)
    A[m] = np.clip(A[m] - a_pull, 0, 255)  # prom no rozā
    return cv2.cvtColor(cv2.merge([L.astype(np.uint8),
                                   A.astype(np.uint8),
                                   B.astype(np.uint8)]), cv2.COLOR_LAB2BGR)

# ── Pipeline ─────────────────────────────────────────────────────────────────
def make_teeth_mask(bgr, landmarks):
    h, w = bgr.shape[:2]
    lips = lips_hull_mask(h, w, landmarks)
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,2]
    low = (np.median(V[lips>0]) < 110)  # sliktā gaisma?

    safe = build_mouth_safe(bgr, landmarks, lowlight=low)
    bgr_eq = clahe_in_mouth(bgr, safe)
    mask = classify_teeth_mahalanobis(bgr_eq, safe, lowlight=low)

    # pārklājuma tests – ja par maz, atslābinām
    area = max(1, np.count_nonzero(safe))
    cov = np.count_nonzero(mask)/area
    if cov < (0.45 if low else 0.4):
        safe2 = dilate(safe, 1, "ellipse")
        mask = classify_teeth_mahalanobis(bgr_eq, safe2, lowlight=True)
    return mask

# ── API ──────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data field 'file'."), 400
        bgr = load_image_fix_orientation(request.files["file"])
        res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422
        landmarks = res.multi_face_landmarks[0].landmark

        mask = make_teeth_mask(bgr, landmarks)
        out = whiten_lab(bgr, mask, l_gain=14, b_shift=22, a_pull=4)

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
