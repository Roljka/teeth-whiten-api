# teeth_api.py
import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Mediapipe (lai /health atbild uzreiz un Render neuzkaras)
_face_mesh = None
_mp = None

def get_face_mesh():
    global _face_mesh, _mp
    if _face_mesh is None:
        import mediapipe as mp
        _mp = mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
    return _face_mesh, _mp

# ─────────────────────────────────────────────────────────────────────────────
# Palīgfunkcijas I/O

def pil_to_bgr(pil_img):
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return pil_to_bgr(img)

# ─────────────────────────────────────────────────────────────────────────────
# Ģeometrija no FaceMesh

def lips_polygon_from_landmarks(h, w, landmarks, mp):
    idx = set()
    for a, b in mp.solutions.face_mesh.FACEMESH_LIPS:
        idx.add(a); idx.add(b)
    pts = []
    for i in idx:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    if pts.shape[0] < 3:
        return None
    hull = cv2.convexHull(pts)
    return hull

def mouth_inner_mask(h, w, hull, shrink_px=2, extra_expand_x=2):
    """ Aizpilda lūpu konveksa čaulu, atvirza no lūpām (šaurāka) un
        nedaudz paplašina horizontāli, lai aizsniegtu sānu zobus. """
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_px+1, 2*shrink_px+1))
        mask = cv2.erode(mask, k, iterations=1)
    if extra_expand_x > 0:
        kx = cv2.getStructuringElement(cv2.MORPH_RECT, (2*extra_expand_x+1, 1))
        mask = cv2.dilate(mask, kx, iterations=1)
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# Krāsu kandidāti (HSV/LAB) + sēklas + reģiona augsme (ΔE)

def percentile(arr, q):
    return float(np.percentile(arr, q)) if arr.size else 0.0

def red_like_mask(hsv, lab):
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)
    red_h = ((H <= 12) | (H >= 170)) & (S > 25)
    red_a = (A > 145)  # “sarkans” LAB A-kanāls
    return (red_h | red_a)

def teeth_candidates(bgr, mouth_mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    mm = mouth_mask > 0
    if np.count_nonzero(mm) == 0:
        return np.zeros_like(mouth_mask), hsv, lab

    # Adaptīvi sliekšņi no mutes iekšpuses
    S_m = S[mm]; V_m = V[mm]; L_m = L[mm]; B_m = B[mm]
    s_thr = min(90, max(60, percentile(S_m, 65)))
    v_thr = max(110, percentile(V_m, 40))
    b_yel = percentile(B_m, 70)  # dzeltenums (jo mazāks → mazāk dzeltens tikai ēna)
    a_thr = percentile(A[mm], 65) + 5

    mouth_inner_cand = (S < s_thr) & (L > percentile(L_m, 30))  # izmanto L pret ēnām
    not_red = ~red_like_mask(hsv, lab)
    not_gums = (A < a_thr + 10)  # atmet “koši sarkano”

    cand = mouth_inner_cand & not_red & not_gums & (mouth_mask > 0)

    # Trokšņu tīrīšana
    m = np.zeros_like(mouth_mask)
    m[cand] = 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)
    return m, hsv, lab

def choose_seeds(hsv, lab, mouth_mask, cand_mask, top_pct=0.7):
    # Sēklas = visgaišākie (V/L) pikseļi zobu kandidātu zonā
    H, S, V = cv2.split(hsv); L, A, B = cv2.split(lab)
    mm = (mouth_mask > 0) & (cand_mask > 0)

    if np.count_nonzero(mm) == 0:
        return np.zeros_like(mouth_mask)

    # rank pēc kombinētas gaišuma metrikas
    score = 0.6*V.astype(np.float32) + 0.4*L.astype(np.float32)
    sel = score[mm]
    if sel.size == 0:
        return np.zeros_like(mouth_mask)

    thr = percentile(sel, 100 - top_pct)  # top X%
    seeds = (score >= thr) & mm

    # atmetam sarkanīgos
    seeds &= ~(red_like_mask(hsv, lab))
    out = np.zeros_like(mouth_mask)
    out[seeds] = 255
    # mazliet salīmēt
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    out = cv2.dilate(out, k3, iterations=1)
    return out

def region_grow_lab(lab, mouth_mask, seeds_mask, delta_e=18.0):
    """ Vienkārša reģiona augsme LAB telpā no sēklām pēc ΔE (CIE76). """
    L, A, B = [ch.astype(np.float32) for ch in cv2.split(lab)]
    h, w = L.shape
    grown = np.zeros((h, w), np.uint8)
    seeds = np.argwhere(seeds_mask > 0)
    if seeds.size == 0:
        return grown

    # sākotnējais vidējais no sēklām
    sm = (seeds_mask > 0)
    L0, A0, B0 = L[sm].mean(), A[sm].mean(), B[sm].mean()

    # BFS
    from collections import deque
    q = deque((int(y), int(x)) for y, x in seeds)
    visited = np.zeros((h, w), np.uint8)
    visited[sm] = 1
    grown[sm] = 255

    while q:
        y, x = q.popleft()
        for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if ny<0 or nx<0 or ny>=h or nx>=w: continue
            if visited[ny, nx]: continue
            visited[ny, nx] = 1
            if mouth_mask[ny, nx] == 0: 
                continue
            # ΔE
            dE = np.sqrt((L[ny,nx]-L0)**2 + (A[ny,nx]-A0)**2 + (B[ny,nx]-B0)**2)
            if dE <= delta_e:
                grown[ny, nx] = 255
                q.append((ny, nx))
    # Pēcaugsme: aizlāpīt caurumus
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    out = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k3, iterations=2)
    return out

def keep_top_components(mask, k=2):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num)]
    areas.sort(reverse=True)
    keep = [idx for _, idx in areas[:k]]
    out = np.zeros_like(mask)
    for i in keep:
        out[labels == i] = 255
    return out

def feather_mask(mask, radius=5):
    if radius <= 0:
        return (mask > 0).astype(np.float32)
    blur = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (0,0), radius)
    return np.clip(blur, 0.0, 1.0).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Galvenā maskas būve + balināšana

def make_teeth_mask(bgr, landmarks):
    h, w = bgr.shape[:2]
    # 1) lūpu čaula → mutes iekšpuse
    _, mp = get_face_mesh()
    hull = lips_polygon_from_landmarks(h, w, landmarks, mp)
    if hull is None:
        return np.zeros((h, w), np.uint8)

    # adaptīvi parametri pēc bildes izmēra
    shrink_px = max(1, min(h, w)//300)
    mouth = mouth_inner_mask(h, w, hull, shrink_px=shrink_px, extra_expand_x=2)

    # 2) kandidāti
    cand, hsv, lab = teeth_candidates(bgr, mouth)

    # 3) sēklas + reģiona augsme (ΔE) ēnām
    seeds = choose_seeds(hsv, lab, mouth, cand, top_pct=0.7)
    grown = region_grow_lab(lab, mouth, seeds, delta_e=18.0)

    # 4) ja par maz – fallback ar vaļīgāku ΔE
    area = int(np.count_nonzero(grown))
    if area < (mouth.sum() / 255) * 0.08:
        grown = region_grow_lab(lab, mouth, seeds, delta_e=26.0)

    # 5) Tīrīšana + top-2 komponentes
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, k3, iterations=1)
    grown = cv2.dilate(grown, k3, iterations=1)
    out = keep_top_components(grown, k=2)

    # 6) noņemam smaganu sarkanos
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gums = red_like_mask(hsv, lab)
    out[gums] = 0

    return out

def whiten_lab(bgr, mask_u8, l_gain=14, b_shift=22, a_pull=4):
    """ Balinām tikai maskā, ar feather malām. """
    if np.count_nonzero(mask_u8) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    alpha = feather_mask(mask_u8, radius=3)  # 0..1

    Lf = L.astype(np.float32)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # TIKAI maskā – izmanto alfa kā intensitāti
    Lf = np.clip(Lf + l_gain * alpha, 0, 255)
    Bf = np.clip(Bf - b_shift * alpha, 0, 255)
    # viegla “neitralizācija” no sarkanā (a* uz 128)
    Af = np.clip(Af + (128.0 - Af) * (a_pull/50.0) * alpha, 0, 255)

    out = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# API

@app.route("/health")
def health():
    return jsonify(ok=True, status="ready")

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data field 'file'."), 400

        bgr = load_image_fix_orientation(request.files["file"])
        face_mesh, _ = get_face_mesh()

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
