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

def shrink(mask, px):
    if px <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.erode(mask, k, 1)

def grow(mask, px):
    if px <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.dilate(mask, k, 1)

def hsv_red_like(bgr, a_thresh):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)
    red = (((H <= 12) | (H >= 170)) & (S > 30)) | (A > a_thresh)
    out = np.zeros(H.shape, np.uint8); out[red] = 255
    return out

def build_mouth_safe(bgr, landmarks, lowlight=False):
    h, w = bgr.shape[:2]
    lips = lips_hull_mask(h, w, landmarks)
    edge = max(1, min(h, w)//240)           # 1–3 px
    inner = shrink(lips, edge)               # atvirzāmies no lūpu robežas

    # “Dziļā mute” – neliela vertikāla dilācija uz leju (smaida apakša)
    kdeep = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5 if lowlight else 4))
    deep = cv2.dilate(inner, kdeep, 1)

    # izmetam rozā/sarkano
    red = hsv_red_like(bgr, a_thresh=(160 if lowlight else 155))
    safe = cv2.bitwise_and(deep, cv2.bitwise_not(red))

    # smooth
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)), 2)

    # fallback
    if np.count_nonzero(safe) < 200:
        safe = inner
    return safe

# ── CLAHE mutē (labāka saderība ēnām) ────────────────────────────────────────
def clahe_in_mouth(bgr, safe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = safe > 0
    if np.count_nonzero(m) == 0:
        return bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = L.copy()
    L2[m] = clahe.apply(L[m])
    out = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    return out

# ── Region grow + “row fill” + distance-close ────────────────────────────────
def grow_teeth_lab(bgr, safe, lowlight):
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]

    m = safe > 0
    if np.count_nonzero(m) < 200:
        return np.zeros((h, w), np.uint8)

    # sēklas: gaišākie, ne-rozā
    seeds = m & (A < (160 if lowlight else 155)) & (V > (85 if lowlight else 95))
    ys, xs = np.where(seeds)
    if xs.size == 0: return np.zeros((h, w), np.uint8)
    idx = np.argsort(-L[ys, xs])[: (350 if lowlight else 250)]
    sy, sx = ys[idx], xs[idx]

    grown = np.zeros((h, w), np.uint8); grown[sy, sx] = 255
    queue = list(zip(sy.tolist(), sx.tolist()))

    # malas: nelaiž pāri asām robežām (smaganu kontūra)
    g = cv2.Laplacian(cv2.GaussianBlur(L.astype(np.uint8),(3,3),0), cv2.CV_16S)
    g = np.abs(g)

    # sākuma centroids
    L0 = int(np.mean(L[sy, sx])); A0 = int(np.mean(A[sy, sx])); B0 = int(np.mean(B[sy, sx]))
    wL, wA, wB = 1.0, 2.4, 1.7
    thr_base = 15.5 if lowlight else 14.5
    thr_step = 0.9 if lowlight else 0.6
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)]

    for it in range(3 if lowlight else 2):
        thr = thr_base + it * thr_step
        head = 0
        while head < len(queue):
            y, x = queue[head]; head += 1
            for dy, dx in nbrs:
                ny, nx = y+dy, x+dx
                if ny<0 or ny>=h or nx<0 or nx>=w: continue
                if grown[ny, nx] or not m[ny, nx]: continue
                if g[ny, nx] > 60: continue
                dL = int(L[ny, nx]) - L0
                dA = int(A[ny, nx]) - A0
                dB = int(B[ny, nx]) - B0
                dist = (wL*(dL*dL) + wA*(dA*dA) + wB*(dB*dB))**0.5
                if dist <= thr:
                    grown[ny, nx] = 255
                    queue.append((ny, nx))
        gy, gx = np.where(grown > 0)
        if gy.size > 200:
            L0 = int(np.median(L[gy, gx]))
            A0 = int(np.median(A[gy, gx]))
            B0 = int(np.median(B[gy, gx]))

    # 1) “row fill” – nepieļaujam pus-zobus
    filled = np.zeros_like(grown)
    rows = np.where(np.sum(safe>0, axis=1) > 0)[0]
    for y in rows:
        cols = np.where(grown[y]>0)[0]
        if cols.size >= 2:
            x0, x1 = cols.min(), cols.max()
            filled[y, x0:x1+1] = 255
    filled = cv2.bitwise_and(filled, safe)

    # 2) distance close – aizver caurumus
    dist = cv2.distanceTransform((filled>0).astype(np.uint8), cv2.DIST_L2, 3)
    filled[dist < 1] = filled[dist < 1]  # noop; saglabā formātu
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)), 2)
    # vertikālais “piespiediens” līdz zobu apakšai/augšai
    ys, xs = np.where(filled>0)
    if xs.size:
        vpush = np.zeros_like(filled)
        for y, x in zip(ys, xs):
            y0 = max(0, y-5 if not lowlight else y-6)
            y1 = min(h-1, y+7 if not lowlight else y+10)
            vpush[y0:y1+1, x] = 255
        filled = cv2.bitwise_and(vpush, safe)

    # pēdējais gludums
    filled = cv2.medianBlur(filled, 3)
    return filled

# ── Whiten ───────────────────────────────────────────────────────────────────
def whiten_lab(bgr, mask, l_gain=14, b_shift=22, a_pull=4):
    if np.count_nonzero(mask) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = mask > 0
    L = L.astype(np.int16); A = A.astype(np.int16); B = B.astype(np.int16)
    L[m] = np.clip(L[m] + l_gain, 0, 255)
    B[m] = np.clip(B[m] - b_shift, 0, 255)
    A[m] = np.clip(A[m] - a_pull, 0, 255)
    out = cv2.cvtColor(cv2.merge([L.astype(np.uint8),
                                  A.astype(np.uint8),
                                  B.astype(np.uint8)]), cv2.COLOR_LAB2BGR)
    return out

# ── Pipeline ─────────────────────────────────────────────────────────────────
def make_teeth_mask(bgr, landmarks):
    h, w = bgr.shape[:2]
    # low-light heuristika
    lips = lips_hull_mask(h, w, landmarks)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]; low = (np.median(V[lips>0]) < 110)

    safe = build_mouth_safe(bgr, landmarks, lowlight=low)
    bgr_eq = clahe_in_mouth(bgr, safe)
    mask = grow_teeth_lab(bgr_eq, safe, lowlight=low)

    # coverage check; ja pārāk maz – atslābinām
    area = max(1, np.count_nonzero(safe))
    cov = np.count_nonzero(mask)/area
    if cov < (0.35 if low else 0.36):
        mask = grow_teeth_lab(bgr_eq, grow(safe, 1), lowlight=True)

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
