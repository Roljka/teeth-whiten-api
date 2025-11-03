import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

# ---------------- Flask ----------------
app = Flask(__name__)
CORS(app)

# ---------------- MediaPipe FaceMesh ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# ---------------- Ielāde + konverti ----------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600) -> np.ndarray:
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_to_bgr(img)

# ---------------- Sejas/mutes ģeometrija ----------------
def lips_points_from_landmarks(h, w, lms):
    """Savācam visus lūpu punktus no FACEMESH_LIPS (unikālie) un projektējam uz (x,y)."""
    idx = set()
    for a, b in mp_face_mesh.FACEMESH_LIPS:
        idx.add(a); idx.add(b)
    pts = []
    for i in idx:
        lm = lms[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    return pts

def mask_from_poly(h, w, pts, convex=True):
    m = np.zeros((h, w), np.uint8)
    if pts is None or len(pts) < 3:
        return m
    if convex:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(m, hull, 255)
    else:
        cv2.fillPoly(m, [pts], 255)
    return m

def build_ceiling(mask):
    """Katram x dod augšējo (mazāko y) balto pikseli; ja nav – -1."""
    h, w = mask.shape
    ceil = np.full(w, -1, dtype=np.int32)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return ceil
    for x in np.unique(xs):
        yvals = ys[xs == x]
        ceil[x] = int(np.min(yvals)) if yvals.size else -1
    return ceil

def build_floor(mask):
    """Katram x dod apakšējo (lielāko y) balto pikseli; ja nav – -1."""
    h, w = mask.shape
    floor = np.full(w, -1, dtype=np.int32)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return floor
    for x in np.unique(xs):
        yvals = ys[xs == x]
        floor[x] = int(np.max(yvals)) if yvals.size else -1
    return floor

def cut_below(mask, floor, lift_px):
    """Nogriez visu ZEM mutes grīdas."""
    out = mask.copy()
    h, w = out.shape
    for x in range(w):
        y = floor[x]
        if y >= 0:
            y1 = min(h, y + lift_px)
            if y1 < h:
                out[y1:h, x] = 0
    return out

def cut_above(mask, ceiling, shave_px):
    """Nogriez visu VIRS mutes griestiem."""
    out = mask.copy()
    h, w = out.shape
    for x in range(w):
        y = ceiling[x]
        if y >= 0:
            y2 = max(0, min(h, y + shave_px))
            if y2 > 0:
                out[0:y2, x] = 0
    return out

# ---------------- Gum guard / lūpu filtrs ----------------
def gums_mask(bgr, mouth_mask, top_guard_px=3, loose=False):
    """Atmetam smaganas un rozā/arkanus toņus mutes iekšpusē."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)

    m = mouth_mask > 0
    out = np.zeros((h, w), np.uint8)

    # Sarkanie/rozā – plašāks low-light gadījumā
    red_like = (((H <= 12) | (H >= 170)) & (S > (25 if loose else 35)))
    rose_like = (A > (148 if loose else 150))  # LAB A kanāls (rozīgums)

    cand = (m & (red_like | rose_like))
    out[cand] = 255

    # Augšējā aizsargjosla (pret smaganām)
    if top_guard_px > 0:
        ceil = build_ceiling(mouth_mask)
        guard = np.zeros_like(out)
        for x in range(w):
            y = ceil[x]
            if y >= 0:
                y2 = min(h, y + top_guard_px)
                guard[y:y2, x] = 255
        out = cv2.bitwise_or(out, guard)

    out = cv2.morphologyEx(out, cv2.MORPH_DILATE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return out

# ---------------- Low-light novērtējums ----------------
def is_lowlight(bgr, safe_mouth):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    m = safe_mouth > 0
    if np.count_nonzero(m) < 200:
        return False
    v_mean = float(np.mean(V[m]))
    v_p40  = float(np.percentile(V[m], 40))
    v_p60  = float(np.percentile(V[m], 60))
    return (v_mean < 135) or (v_p60 - v_p40 < 12)

# ---------------- Masku būve ----------------
def build_masks(bgr, lms, lowlight=False):
    h, w = bgr.shape[:2]
    lip_pts = lips_points_from_landmarks(h, w, lms)
    outer = mask_from_poly(h, w, lip_pts, convex=True)       # lūpu ārējā zona

    # "Iekšējā mute" – paplašināta/šūta versija ap zobiem
    inner = cv2.erode(outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,7)), 1)
    inner_wide = cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_RECT,(37,11)), 1)

    lips_only = cv2.subtract(outer, inner)

    floor = build_floor(outer)
    ceil = build_ceiling(outer)
    inner_wide = cut_below(inner_wide, floor, lift_px=1)
    inner_wide = cut_above(inner_wide, ceil, shave_px=(5 if lowlight else 3))

    gum = gums_mask(bgr, inner_wide, top_guard_px=(5 if lowlight else 3), loose=lowlight)
    safe_mouth = cv2.subtract(inner_wide, gum)     # mute bez smaganām
    safe_mouth = cv2.subtract(safe_mouth, lips_only)  # un bez lūpām

    return outer, inner_wide, lips_only, safe_mouth

# ---------------- Teet h mask – krāsu soļi ----------------
def teeth_from_colors(bgr, safe_mouth, lowlight=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)

    if not lowlight:
        cand = ((S < 90) & (V > 145) & (A < 150) & (safe_mouth > 0))
    else:
        cand = ((S < 120) & (V > 110) & (A < 158) & (safe_mouth > 0))

    out = np.zeros_like(safe_mouth)
    out[cand] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return out

def teeth_adaptive_percentiles(bgr, safe_mouth):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); _,A,_ = cv2.split(lab)
    m = safe_mouth > 0
    out = np.zeros_like(safe_mouth)
    if np.count_nonzero(m) == 0:
        return out
    s_th = np.percentile(S[m], 65)   # relaksētāks
    v_th = np.percentile(V[m], 52)
    a_th = np.percentile(A[m], 68)
    cand = ((S < s_th) & (V > v_th) & (A < a_th) & m)
    out[cand] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return out

def teeth_adaptive_contrast(bgr, safe_mouth):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Vc = clahe.apply(V)
    Vc = cv2.bitwise_and(Vc, Vc, mask=safe_mouth)
    # Otsu tikai mutē
    _, otsu = cv2.threshold(Vc[ safe_mouth>0 ], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tmp = np.zeros_like(safe_mouth)
    tmp[ safe_mouth>0 ] = (Vc[ safe_mouth>0 ] >= otsu).astype(np.uint8)*255
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return tmp

# ---------------- Maskas izlīdzināšana ----------------
def fill_rows(mask, safe_mouth):
    """Katru rindu aizpilda starp pirmo/pēdējo balto – tikai mutē."""
    h, w = mask.shape
    out = mask.copy()
    rows = np.where(safe_mouth.sum(1) > 0)[0]
    for y in rows:
        xs = np.where(mask[y] > 0)[0]
        if xs.size >= 2:
            x1, x2 = xs.min(), xs.max()
            seg = safe_mouth[y, x1:x2+1] > 0
            out[y, x1:x2+1] = np.where(seg, 255, out[y, x1:x2+1])
    return out

def expand_cols_vertical(teeth, safe_mouth, up=3, down=4):
    h, w = teeth.shape
    out = teeth.copy()
    cols = np.where(safe_mouth.sum(0) > 0)[0]
    for x in cols:
        ys = np.where(teeth[:, x] > 0)[0]
        if ys.size >= 2:
            y1, y2 = ys.min(), ys.max()
            y1 = max(0, y1 - up); y2 = min(h - 1, y2 + down)
            col_ok = safe_mouth[:, x] > 0
            out[y1:y2+1, x] = np.where(col_ok[y1:y2+1], 255, out[y1:y2+1, x])
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5)), 1)
    return out

# ---------------- Kāpņu loģika ----------------
def make_teeth_mask(bgr, lms):
    # novērtējam gaismu
    _, _, _, safe_base = build_masks(bgr, lms, lowlight=False)
    low = is_lowlight(bgr, safe_base)

    outer, inner_wide, lips_only, safe = build_masks(bgr, lms, lowlight=low)

    # 1) krāsu sliekšņi
    t = teeth_from_colors(bgr, safe, lowlight=low)
    t = fill_rows(t, safe)
    t = expand_cols_vertical(t, safe, up=(4 if low else 3), down=(6 if low else 4))
    t = cv2.bitwise_and(t, safe)
    area = max(1, np.count_nonzero(safe))
    cov = np.count_nonzero(t)
    if cov/area >= (0.32 if low else 0.34):
        return t

    # 2) adaptīvie procentili
    t = teeth_adaptive_percentiles(bgr, safe)
    t = fill_rows(t, safe)
    t = expand_cols_vertical(t, safe, up=(4 if low else 3), down=(6 if low else 4))
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t)
    if cov/area >= (0.30 if low else 0.33):
        return t

    # 3) lokāls kontrasts
    t = teeth_adaptive_contrast(bgr, safe)
    t = fill_rows(t, safe)
    t = expand_cols_vertical(t, safe, up=(5 if low else 3), down=(7 if low else 4))
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t)
    if cov/area >= (0.28 if low else 0.30):
        return t

    # 4) glābiņš – visa drošā mute ar 1px skūšanu
    ceil = build_ceiling(safe); floor = build_floor(safe)
    t = safe.copy()
    t = cut_above(t, ceil, shave_px=1)
    t = cut_below(t, floor, lift_px=1)
    return t

# ---------------- Balināšana (ar feather) ----------------
def whiten_only_teeth(bgr: np.ndarray, mask: np.ndarray,
                      l_gain: int = 10, b_shift: int = 18) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return bgr

    # 1–2 px feather mala
    alpha = cv2.GaussianBlur((mask/255.0).astype(np.float32), (3,3), 0)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Ln = L.astype(np.float32)
    Bn = B.astype(np.float32)

    Ln = np.clip(Ln + l_gain * alpha, 0, 255).astype(np.uint8)
    Bn = np.clip(Bn - b_shift * alpha, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(cv2.merge([Ln, A, Bn]), cv2.COLOR_LAB2BGR)
    return out

# ---------------- API ----------------
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data with field 'file'."), 400

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        lms = res.multi_face_landmarks[0].landmark

        # Zobu maska (ar low-light kāpnēm)
        teeth_mask = make_teeth_mask(bgr, lms)

        # Balinām tikai maskā
        out = whiten_only_teeth(bgr, teeth_mask, l_gain=12, b_shift=20)

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
