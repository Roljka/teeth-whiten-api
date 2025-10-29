import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

# ------------------------------
# Flask
# ------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------
# Mediapipe FaceMesh (viegls režīms)
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,     # nav vajadzīgi iris u.c. punkti
    min_detection_confidence=0.5
)

# ------------------------------
# Palīgfunkcijas
# ------------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Pillow -> OpenCV BGR"""
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600) -> np.ndarray:
    """
    Nolasa bildi, salabo EXIF rotāciju un, ja vajag, samazina izmēru
    (taupa RAM/CPU uz hosta).
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_to_bgr(img)

def lips_outer_mask(h: int, w: int, landmarks) -> np.ndarray:
    """
    Mutes ārējās lūpas polygon-mask (no FACEMESH_LIPS).
    Pēc tam to “ieēdam” uz iekšu, lai iegūtu iekšējo mutes zonu.
    """
    idx = set()
    for a, b in mp_face_mesh.FACEMESH_LIPS:
        idx.add(a); idx.add(b)

    pts = []
    for i in idx:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    if len(pts) < 3:
        return np.zeros((h, w), np.uint8)

    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Pagriežam šo par "inner-lip" aproksimāciju ar eroziju.
    # Kernel izmēru pielāgojam mutes augstumam (~1.5–2% no īsākās malas)
    k = max(2, int(min(h, w) * 0.015))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return mask

def build_teeth_mask_adaptive(bgr: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
    """
    Adaptīva zobu maska, strādā tikai iekš mutes (inner_mask):
      1) CLAHE normalizē L (LAB) tikai ROI → tumšie zobi paceļas
      2) score = Lnorm - 0.6*Snorm + 0.4*(255 - B)
      3) Otsu slieksnis tikai ROI
      4) Morfoloģija, aizlāpīt starp zobiem
      5) Izmetam lūpu/smaganu komponentes pēc S/A vidējām vērtībām
    """
    if inner_mask is None or np.count_nonzero(inner_mask) == 0:
        return np.zeros(bgr.shape[:2], np.uint8)

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    roi = inner_mask > 0

    # 1) CLAHE uz L tikai ROI
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = L.copy()
    L_eq[roi] = clahe.apply(L[roi])

    # 2) score
    Ln = cv2.normalize(L_eq, None, 0, 255, cv2.NORM_MINMAX)
    Sn = cv2.normalize(S,    None, 0, 255, cv2.NORM_MINMAX)
    score = (Ln.astype(np.float32)
             - 0.6 * Sn.astype(np.float32)
             + 0.4 * (255.0 - B.astype(np.float32)))
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3) Otsu tikai ROI pikseļiem
    vals = score[roi]
    if vals.size < 50:
        return np.zeros_like(inner_mask)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros_like(inner_mask)
    mask[(score >= thr) & roi] = 255

    # 4) Morfo – aizlīmē spraugas starp kroziņām
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

    # 5) Izmetam rozīgus/sat. komponentus (lūpas/smaganas)
    num, labels = cv2.connectedComponents(mask)
    out = np.zeros_like(mask)
    for lid in range(1, num):
        comp = (labels == lid)
        if comp.sum() < 60:   # ļoti mazas daļas laukā
            continue
        meanS = float(S[comp].mean())
        meanA = float(A[comp].mean())
        if meanS > 110 or meanA > 145:
            continue
        out[comp] = 255

    return out

def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 14, b_shift: int = 22) -> np.ndarray:
    """
    Balināšana LAB telpā tikai maskā:
      - palielinām L (gaišumu)
      - samazinām B (dzeltenumu)
    """
    if teeth_mask is None or np.count_nonzero(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask = teeth_mask > 0

    Ln = L.astype(np.int16); Bn = B.astype(np.int16)
    Ln[mask] = np.clip(Ln[mask] + l_gain, 0, 255)
    Bn[mask] = np.clip(Bn[mask] - b_shift, 0, 255)

    out = cv2.cvtColor(cv2.merge([Ln.astype(np.uint8), A, Bn.astype(np.uint8)]),
                       cv2.COLOR_LAB2BGR)
    return out

# ------------------------------
# API
# ------------------------------
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: send multipart/form-data with field 'file'."), 400

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        # FaceMesh → lūpu/mutes zona
        results = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = results.multi_face_landmarks[0].landmark
        inner_mask = lips_outer_mask(h, w, landmarks)                 # mutes iekšējā zona (aproksimēta)
        teeth_mask = build_teeth_mask_adaptive(bgr, inner_mask)       # adaptīva zobu maska
        out = whiten_only_teeth(bgr, teeth_mask, l_gain=14, b_shift=22)

        # JPEG outs
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify(error="Encoding failed"), 500

        return send_file(io.BytesIO(buf.tobytes()),
                         mimetype="image/jpeg",
                         as_attachment=False,
                         download_name="whitened.jpg")

    except Exception as e:
        return jsonify(error=str(e)), 500

# ------------------------------
# Dev serveris (Render komandā lieto gunicorn)
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
