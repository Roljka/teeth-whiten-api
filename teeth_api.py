import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

# ----------------------------
# Flask + CORS
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# MediaPipe FaceMesh (viegls, statisks režīms)
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# ----------------------------
# Palīgfunkcijas
# ----------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Pillow -> OpenCV (BGR)."""
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600) -> np.ndarray:
    """
    Nolasa bildi, salabo EXIF rotāciju un samazina max malu (RAM/CPU taupīšanai).
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_to_bgr(img)

def lips_mask_from_landmarks(h: int, w: int, landmarks) -> np.ndarray:
    """
    Uztaisa mutes/lūpu masku no 468 sejas punktiem (FACEMESH_LIPS convex hull).
    """
    lips_connections = mp_face_mesh.FACEMESH_LIPS
    idx = set()
    for a, b in lips_connections:
        idx.add(a)
        idx.add(b)
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

def safe_inner(inner_roi: np.ndarray) -> np.ndarray:
    """
    Drošs mutes iekšējais reģions: attālināmies no lūpu robežas, lai balināšana
    nekad neaiziet līdz lūpām.
    """
    dist = cv2.distanceTransform((inner_roi > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dmin = max(2, int(round(0.004 * max(inner_roi.shape))))
    safe = (dist > dmin).astype(np.uint8) * 255
    return safe

def mad(x: np.ndarray):
    """Median Absolute Deviation + mediāna (numeriskai robustai sliekšņošanai)."""
    x = x.astype(np.float32)
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-6, med

def build_teeth_mask_adaptive(bgr: np.ndarray, inner_roi: np.ndarray) -> np.ndarray:
    """
    Precīza zobu maska:
      - izmanto drošu mutes iekšpusi (distance transform);
      - sēklas no pašiem baltākajiem pikseļiem + geodēziska izaugsme, lai paņem arī tumšākus zobus;
      - lūpu/smaganu sarkanīgā reģiona supresija ar Cr/H/S;
      - morfoloģiska tīrīšana un atkāpšanās no robežām.
    """
    h, w = bgr.shape[:2]
    if np.count_nonzero(inner_roi) == 0:
        return np.zeros((h, w), np.uint8)

    inner = safe_inner(inner_roi)
    roi = inner > 0

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # "whiteness" metriks – augsts L, zems B, zems S
    whiteness = L.astype(np.float32) - 0.55 * B.astype(np.float32) - 0.25 * S.astype(np.float32)

    # Statistika tikai mutē
    w_roi = whiteness[roi]
    if w_roi.size == 0:
        return np.zeros((h, w), np.uint8)

    S_roi, Cr_roi, H_roi, L_roi, B_roi = S[roi], Cr[roi], H[roi], L[roi], B[roi]
    mad_w, med_w = mad(w_roi)
    mad_S, med_S = mad(S_roi)
    mad_Cr, med_Cr = mad(Cr_roi)
    mad_L, med_L = mad(L_roi)
    mad_B, med_B = mad(B_roi)

    # Lūpu supresija
    lips_red = (Cr > (med_Cr + 1.1 * mad_Cr)) | ( ((H < 15) | (H > 165)) & (S > med_S) )
    lips_red = lips_red & roi

    # Sēklas – pašas baltākās vietas mutē, zems piesātinājums, nav sarkanā
    seed_thr = np.percentile(w_roi, 78)
    seeds = (whiteness > seed_thr) & (S < (med_S + 0.8 * mad_S)) & roi & (~lips_red)

    # Izaugsme – ļaujam nedaudz tumšākus, ja tie ir “balti pietiekami” un mutē
    grow_thr = seed_thr - (0.8 * mad_w + 4)
    allow = (whiteness > grow_thr) & (B < (med_B + 1.0 * mad_B)) & (L > (med_L - 0.5 * mad_L)) & roi & (~lips_red)

    # Geodēziska dilatācija (dilatē tikai atļautajos)
    grow = seeds.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(4):
        cand = cv2.dilate(grow.astype(np.uint8) * 255, kernel, iterations=1) > 0
        grow = (cand & allow) | seeds

    mask = (grow.astype(np.uint8) * 255)

    # Tīrīšana + neliela atkāpšanās no smaganu malas
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    min_area = max(60, int(0.0005 * np.count_nonzero(roi)))  # nelielu troksni ārā
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    out = cv2.erode(out, kernel, iterations=1)

    return out

def whiten_lab(bgr: np.ndarray, mask: np.ndarray, strength: float = 1.05) -> np.ndarray:
    """
    Balināšana LAB telpā tikai maskā:
      - adaptīvs L pieaugums un b* samazinājums no maskas statistikas;
      - feather (Gaussian) mīkstām malām;
      - ārpus maskas attēls paliek nemainīts.
    """
    if np.count_nonzero(mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = (mask > 0)

    Bm = float(np.mean(B[m])) if np.any(m) else 150.0
    Lm = float(np.mean(L[m])) if np.any(m) else 140.0
    b_shift = int(np.clip((Bm - 135.0) * 0.6, 12, 26))
    l_gain  = int(np.clip((150.0 - Lm) * 0.5, 12, 22))

    soft = cv2.GaussianBlur((m.astype(np.uint8) * 255), (0, 0), sigmaX=1.5, sigmaY=1.5)
    soft = (soft.astype(np.float32) / 255.0) * np.clip(strength, 0.5, 1.5)

    Lf = L.astype(np.float32)
    Bf = B.astype(np.float32)
    L_new = np.clip(Lf + soft * l_gain, 0, 255)
    B_new = np.clip(Bf - soft * b_shift, 0, 255)

    out_lab = cv2.merge([L_new.astype(np.uint8), A, B_new.astype(np.uint8)])
    out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    out[~m] = bgr[~m]
    return out

# ----------------------------
# API Endpoints
# ----------------------------
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    """
    Pieņem multipart/form-data { file: <image> }
    Atgriež JPEG ar balinātiem zobiem.
    """
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data with field 'file'."), 400

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        # FaceMesh (RGB)
        results = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = results.multi_face_landmarks[0].landmark
        lips_mask = lips_mask_from_landmarks(h, w, landmarks)

        # Precīza zobu maska un balināšana
        teeth_mask = build_teeth_mask_adaptive(bgr, lips_mask)
        out = whiten_lab(bgr, teeth_mask, strength=1.08)

        # JPEG atmiņā
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify(error="Encode failed"), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="whitened.jpg"
        )

    except Exception as e:
        return jsonify(error=str(e)), 500

# ----------------------------
# Local run
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
