import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp
from sklearn.cluster import KMeans

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

# ---------- Palīgfunkcijas ----------
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

# Iekšējās lūpas (inner lips) indeksi no FaceMesh (aptuvena kopa, pietiekami stabila)
INNER_LIP_IDX = sorted({
    0, 11, 12, 13, 14, 15, 16, 61, 62, 63, 64, 65, 66, 67,
    78, 79, 80, 81, 82, 87, 88, 89, 90, 91, 95,
    146, 164, 178, 191, 267, 268, 269, 270, 271, 272, 273, 274,
    275, 291, 292, 293, 294, 295, 296, 297, 313, 314, 315, 324,
    375, 402, 405, 409, 415, 417
})

def poly_mask_from_indices(h, w, landmarks, idx_set) -> np.ndarray:
    pts = []
    for i in idx_set:
        if i < len(landmarks):
            lm = landmarks[i]
            pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask

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

def build_teeth_mask_adaptive(bgr: np.ndarray, inner_lip_mask: np.ndarray) -> np.ndarray:
    """Adaptīva zobu maska mutes iekšienē (LAB/HSV/YCrCb + KMeans)."""
    h, w = bgr.shape[:2]
    roi = inner_lip_mask > 0
    if np.count_nonzero(roi) == 0:
        return np.zeros((h, w), np.uint8)

    # Konverti
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Statistika tikai mutes iekšā
    L_roi = L[roi]; B_roi = B[roi]; S_roi = S[roi]; Cr_roi = Cr[roi]

    mad_L, med_L = mad(L_roi)
    mad_B, med_B = mad(B_roi)
    mad_S, med_S = mad(S_roi)
    mad_Cr, med_Cr = mad(Cr_roi)

    # Adaptīvi sliekšņi (konservatīvi)
    kL, kB, kS, kCr = 0.8, 0.6, 1.2, 0.8
    cond_bright   = L > (med_L + kL * mad_L)
    cond_less_yel = B < (med_B - kB * mad_B)
    cond_not_sat  = S < (med_S + kS * mad_S)

    # Izslēdzam lūpas pēc "sarkanuma"
    lips_red = Cr > (med_Cr + kCr * mad_Cr)

    base = cond_bright & cond_less_yel & cond_not_sat & roi & (~lips_red)

    # KMeans (k=3) uz [L, B] tikai ROI
    ys, xs = np.where(roi)
    feat = np.stack([L[ys, xs].astype(np.float32), B[ys, xs].astype(np.float32)], axis=1)
    try:
        km = KMeans(n_clusters=3, n_init=5, random_state=0)
        labels = km.fit_predict(feat)
        centers = km.cluster_centers_
        # izvelkam klasi ar max L un min B (balti, maz dzeltena)
        score = centers[:, 0] - centers[:, 1]  # augsta L - zema B
        best = np.argmax(score)
        km_mask = np.zeros((h, w), np.uint8)
        km_mask[ys, xs] = (labels == best).astype(np.uint8) * 255
        cand = (base | (km_mask > 0)) & roi & (~lips_red)
    except Exception:
        cand = base

    # Morfoloģija + komponentes
    kernel = np.ones((3, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8); mask[cand] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    min_area = max(40, int(0.0005 * np.count_nonzero(roi)))  # relatīvs slieksnis
    mask = remove_small_components(mask, min_area)
    # Mazliet erode, lai nenobrauktu smaganu robežu
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def whiten_lab(bgr: np.ndarray, mask: np.ndarray,
               l_gain=16, b_shift=20, alpha=0.85) -> np.ndarray:
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

    # soft blend tikai maskā (lai nav “tīrs plāksteris”)
    out = (out.astype(np.float32) * alpha + bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    out[~m] = bgr[~m]
    return out

# ---------- API ----------
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

        # FaceMesh
        res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422
        lm = res.multi_face_landmarks[0].landmark

        inner_mask = poly_mask_from_indices(h, w, lm, INNER_LIP_IDX)
        if np.count_nonzero(inner_mask) == 0:
            return jsonify(error="Mouth not detected"), 422

        teeth_mask = build_teeth_mask_adaptive(bgr, inner_mask)
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
