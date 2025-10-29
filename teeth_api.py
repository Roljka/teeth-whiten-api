import io, os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ---- Mediapipe FaceMesh (viegls, 1 seja) ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

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

# ---------- MUTES MASKU BŪVE (droša) ----------
def lips_outer_mask(h, w, landmarks) -> (np.ndarray, tuple):
    """Hull no FACEMESH_LIPS punktiem + drošības bbox (apgriež ROI)."""
    idx = set()
    for a, b in mp_face_mesh.FACEMESH_LIPS:
        idx.add(a); idx.add(b)
    pts = []
    for i in idx:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        x, y, bw, bh = cv2.boundingRect(hull)
        # drošības rezerve ap muti (ne vairāk par attēla robežām)
        pad = int(round(max(bw, bh) * 0.4))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
        bbox = (x0, y0, x1, y1)
        return mask, bbox
    return mask, (0, 0, w, h)

def mouth_inner_mask(outer_mask: np.ndarray) -> np.ndarray:
    """Iegūst iekšējo mutes zonu (ne skart lūpas): erode atkarībā no mutes izmēra."""
    area = int(np.count_nonzero(outer_mask))
    # kodola izmēru balstu pret mutes laukumu (robusti dažādiem attēliem)
    k = max(2, int(round(np.sqrt(max(area,1)) / 30)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(outer_mask, kernel, iterations=1)
    return inner

def mad(x):
    x = x.astype(np.float32)
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-6, med

def remove_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def build_teeth_mask_adaptive(bgr: np.ndarray, inner_roi: np.ndarray) -> np.ndarray:
    """Adaptīva zobu maska mutes iekšienē (LAB/HSV/YCrCb + cv2.kmeans)."""
    h, w = bgr.shape[:2]
    roi = inner_roi > 0
    if np.count_nonzero(roi) == 0:
        return np.zeros((h, w), np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, _ = cv2.split(ycrcb)

    Lr, Br, Sr, Crr = L[roi], B[roi], S[roi], Cr[roi]
    mad_L, med_L = mad(Lr); mad_B, med_B = mad(Br)
    mad_S, med_S = mad(Sr); mad_Cr, med_Cr = mad(Crr)

    kL, kB, kS, kCr = 0.8, 0.6, 1.2, 1.0  # lips supresijai mazliet stingrāks Cr
    cond_bright   = L > (med_L + kL * mad_L)
    cond_less_yel = B < (med_B - kB * mad_B)
    cond_not_sat  = S < (med_S + kS * mad_S)
    lips_red      = Cr > (med_Cr + kCr * mad_Cr)

    base = cond_bright & cond_less_yel & cond_not_sat & roi & (~lips_red)

    # --- cv2.kmeans (bez sklearn) uz ROI [L, B] ---
    ys, xs = np.where(roi)
    Z = np.stack([L[ys, xs].astype(np.float32), B[ys, xs].astype(np.float32)], axis=1)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        K = 3
        _ret, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.float32)  # [K,2] -> [L,B]
        score = centers[:, 0] - centers[:, 1]  # augsts L, zems B
        best = int(np.argmax(score))
        km_mask = np.zeros((h, w), np.uint8)
        km_mask[ys, xs] = (labels.flatten() == best).astype(np.uint8) * 255
        cand = (base | (km_mask > 0)) & roi & (~lips_red)
    except Exception:
        cand = base

    kernel = np.ones((3, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8); mask[cand] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # atmet mazās pikseļu salas
    min_area = max(50, int(0.0006 * np.count_nonzero(roi)))
    mask = remove_small_components(mask, min_area)
    # nedaudz atvirzāmies no līnijām uz smaganām/lūpām
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def whiten_lab(bgr: np.ndarray, mask: np.ndarray, l_gain=18, b_shift=22, alpha=0.9) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    m = mask > 0
    L2 = L.astype(np.int16); B2 = B.astype(np.int16)
    L2[m] = np.clip(L2[m] + l_gain, 0, 255)
    B2[m] = np.clip(B2[m] - b_shift, 0, 255)
    out_lab = cv2.merge([L2.astype(np.uint8), A, B2.astype(np.uint8)])
    out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    out = (out.astype(np.float32) * alpha + bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    out[~m] = bgr[~m]
    return out

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
        lm = res.multi_face_landmarks[0].landmark

        # 1) Lūpu ārējā maska + bbox
        outer_lips, (x0, y0, x1, y1) = lips_outer_mask(h, w, lm)
        if np.count_nonzero(outer_lips) == 0:
            return jsonify(error="Mouth not detected"), 422

        # 2) Iekšējā mutes maska (ne skart lūpas)
        inner = mouth_inner_mask(outer_lips)

        # 3) Stingrs ROI – nogriežam visu ārpus drošības bbox
        bbox_mask = np.zeros((h, w), np.uint8)
        bbox_mask[y0:y1, x0:x1] = 255
        inner_roi = cv2.bitwise_and(inner, bbox_mask)

        # 4) Zobu maska + balināšana
        teeth_mask = build_teeth_mask_adaptive(bgr, inner_roi)
        out = whiten_lab(bgr, teeth_mask, l_gain=18, b_shift=22, alpha=0.9)

        _, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return send_file(io.BytesIO(buf.tobytes()),
                         mimetype="image/jpeg",
                         as_attachment=False,
                         download_name="whitened.jpg")
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
