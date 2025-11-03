import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Mediapipe FaceMesh – 1 seja, statisks, pietiek ar lūpām
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# ──────────────────────────────────────────────────────────────────────────────
# Palīgfunkcijas (I/O)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Mutes zonas veidošana no FaceMesh lūpu punktiem
# ──────────────────────────────────────────────────────────────────────────────
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

def build_mouth_masks(bgr, landmarks, lowlight=False):
    h, w = bgr.shape[:2]
    lips = lips_hull_mask(h, w, landmarks)

    # “Drošā mute”: lūpu huls + neliela eroze no robežām
    edge = max(1, min(h, w)//240)     # 1–3 px tipiski
    safe = shrink(lips, edge)

    # Atmetam rozā (smaganas/lūpas) pēc LAB A kanāla un HSV sarkanā
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    red_like = (((H <= 12) | (H >= 170)) & (S > 30)) | (A > (155 if lowlight else 150))
    lips_color = np.zeros((h, w), np.uint8); lips_color[red_like] = 255

    # safe_mouth = lips iekšpuse bez rozā pikseļiem + neliela aizpilde
    safe_mouth = cv2.bitwise_and(safe, cv2.bitwise_not(lips_color))
    safe_mouth = cv2.morphologyEx(safe_mouth, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    safe_mouth = cv2.morphologyEx(safe_mouth, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)), 2)

    # Drošības tīkls – ja paliek par maz, ņemam lips bez krāsas maskas
    if np.count_nonzero(safe_mouth) < 200:
        safe_mouth = safe

    return safe_mouth

# ──────────────────────────────────────────────────────────────────────────────
# Krāsu-līdzības reģionu augsme (LAB) ZOBU maskai
# ──────────────────────────────────────────────────────────────────────────────
def teeth_region_grow_lab(bgr, safe_mouth,
                          k_seeds=250, a_max=155, v_min=95,
                          wL=1.0, wA=2.2, wB=1.6,
                          thr_base=14.5, thr_step=0.6, max_iter=2):
    """
    Aug reģionu no gaišākajiem “ne-rozā” pikseļiem mutes iekšpusē.
    Distances telpa: sqrt(wL*dL^2 + wA*dA^2 + wB*dB^2).
    """
    h, w = bgr.shape[:2]
    if np.count_nonzero(safe_mouth) < 200:
        return np.zeros((h, w), np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]

    m = safe_mouth > 0
    seeds_mask = (m & (A < a_max) & (V > v_min))

    ys, xs = np.where(seeds_mask)
    if xs.size == 0:
        return np.zeros((h, w), np.uint8)
    idx = np.argsort(-L[ys, xs])[:k_seeds]  # spilgtākie
    sy, sx = ys[idx], xs[idx]

    grown = np.zeros((h, w), np.uint8)
    grown[sy, sx] = 255
    queue = list(zip(sy.tolist(), sx.tolist()))

    # Nelaižam cauri asām malām
    g = cv2.Laplacian(cv2.GaussianBlur(L.astype(np.uint8),(3,3),0), cv2.CV_16S)
    g = np.abs(g)

    L0 = int(np.mean(L[sy, sx])); A0 = int(np.mean(A[sy, sx])); B0 = int(np.mean(B[sy, sx]))
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)]

    for it in range(max_iter):
        thr = thr_base + it * thr_step
        head = 0
        while head < len(queue):
            y, x = queue[head]; head += 1
            for dy, dx in nbrs:
                ny, nx = y+dy, x+dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if grown[ny, nx] or not m[ny, nx]:
                    continue
                if g[ny, nx] > 60:  # asa mala (smaganu robeža u.tml.)
                    continue
                dL = int(L[ny, nx]) - L0
                dA = int(A[ny, nx]) - A0
                dB = int(B[ny, nx]) - B0
                dist = (wL*(dL*dL) + wA*(dA*dA) + wB*(dB*dB))**0.5
                if dist <= thr:
                    grown[ny, nx] = 255
                    queue.append((ny, nx))

        gy, gx = np.where(grown > 0)
        if gy.size > 200:
            # robustāk pret ēnām – median
            L0 = int(np.median(L[gy, gx]))
            A0 = int(np.median(A[gy, gx]))
            B0 = int(np.median(B[gy, gx]))

    # Nedaudz izlīdzinām
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)), 1)
    return grown

# “Smaida aizvēršana”: aizpilda atstarpes pa rindām un vertikāli izstiepj
def fill_rows(mask, safe_mouth):
    h, w = mask.shape
    out = np.zeros_like(mask)
    rows = np.where(np.sum(safe_mouth>0, axis=1) > 0)[0]
    if rows.size == 0: 
        return mask
    y0, y1 = rows[0], rows[-1]
    for y in range(y0, y1+1):
        cols = np.where(mask[y]>0)[0]
        if cols.size >= 2:
            out[y, cols.min():cols.max()+1] = 255
    return out

def expand_cols_vertical(mask, safe_mouth, up=4, down=6):
    h, w = mask.shape
    ys, xs = np.where(mask>0)
    if xs.size == 0:
        return mask
    out = np.zeros_like(mask)
    for y, x in zip(ys, xs):
        y0 = max(0, y-up); y1 = min(h-1, y+down)
        out[y0:y1+1, x] = 255
    out = cv2.bitwise_and(out, safe_mouth)
    out = cv2.medianBlur(out, 3)
    return out

# Galvenā zobu maska
def make_teeth_mask(bgr, landmarks):
    safe_base = build_mouth_masks(bgr, landmarks, lowlight=False)
    area = max(1, np.count_nonzero(safe_base))
    # low-light heuristika
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]; low = (np.median(V[safe_base>0]) < 110)

    safe = build_mouth_masks(bgr, landmarks, lowlight=low)

    # 0) krāsu-līdzības augsme
    rg = teeth_region_grow_lab(
        bgr, safe,
        k_seeds=(350 if low else 250),
        a_max=(160 if low else 155),
        v_min=(85 if low else 95),
        wL=1.0, wA=2.4, wB=1.7,
        thr_base=(15.5 if low else 14.5),
        thr_step=(0.9 if low else 0.6),
        max_iter=(3 if low else 2)
    )
    t = fill_rows(rg, safe)
    t = expand_cols_vertical(t, safe, up=(5 if low else 3), down=(7 if low else 4))
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t)/area
    if cov >= (0.35 if low else 0.36):
        return t

    # 1) ja nepietiek – atslābinām vertikālo izplešanu, lai “aizsniedz dibenu”
    t = fill_rows(rg, safe)
    t = expand_cols_vertical(t, safe, up=(6 if low else 4), down=(10 if low else 6))
    t = cv2.bitwise_and(t, safe)
    return t

# ──────────────────────────────────────────────────────────────────────────────
# Balināšana LAB telpā tikai maskā
# ──────────────────────────────────────────────────────────────────────────────
def whiten_lab(bgr, mask, l_gain=14, b_shift=22, a_pull=4):
    if np.count_nonzero(mask) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = mask > 0
    L = L.astype(np.int16); A = A.astype(np.int16); B = B.astype(np.int16)
    L[m] = np.clip(L[m] + l_gain, 0, 255)
    B[m] = np.clip(B[m] - b_shift, 0, 255)   # mazāk dzeltena
    # mazliet “noņemam” rozā tonējumu zobos
    A[m] = np.clip(A[m] - a_pull, 0, 255)
    out = cv2.cvtColor(cv2.merge([L.astype(np.uint8),
                                  A.astype(np.uint8),
                                  B.astype(np.uint8)]), cv2.COLOR_LAB2BGR)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data field 'file'."), 400

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark

        teeth_mask = make_teeth_mask(bgr, landmarks)
        out = whiten_lab(bgr, teeth_mask, l_gain=14, b_shift=22, a_pull=4)

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
