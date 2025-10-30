import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ---------- Mediapipe ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# cik daudz mutes mums jāaizpilda, lai teiktu "ok"
MIN_RATIO_OK = 0.30   # 30% no mutes iekšpuses
MIN_PX_OK = 350

def load_image_fix_orientation(file_storage, max_side=1600) -> np.ndarray:
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def enhance_for_detection(bgr: np.ndarray) -> np.ndarray:
    gamma = 1.1
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    table = table.astype("uint8")
    bgr_gamma = cv2.LUT(bgr, table)

    lab = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def lips_mask_from_landmarks(h, w, landmarks) -> np.ndarray:
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

def shrink_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.erode(mask, k, iterations=1)

def expand_hv(mask: np.ndarray, mouth_inner: np.ndarray, hx: int = 20, vy: int = 5) -> np.ndarray:
    """spēcīgi izplešam pa x un mazliet pa y, bet tikai mutes iekšpusē"""
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (hx, 3))
    mx = cv2.dilate(mask, kx, iterations=1)
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vy))
    mxy = cv2.dilate(mx, ky, iterations=1)
    out = np.zeros_like(mask)
    out[(mxy > 0) & (mouth_inner > 0)] = 255
    return out

def keep_top_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = []
    for i in range(1, num_labels):
        areas.append((stats[i, cv2.CC_STAT_AREA], i))
    areas.sort(reverse=True)
    keep = [idx for (_, idx) in areas[:n]]
    out = np.zeros_like(mask)
    for i in keep:
        out[labels == i] = 255
    return out

# ---------- 1) HSV līmenis ----------
def teeth_mask_hsv(bgr: np.ndarray, mouth_inner: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    cand = (S < 105) & (V > 135) & (mouth_inner > 0)
    red_like = (((H <= 12) | (H >= 170)) & (S > 30))
    cand = cand & (~red_like)

    mask = np.zeros((h, w), np.uint8)
    mask[cand] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    return mask

# ---------- 2) adaptīvais pa pusēm ----------
def teeth_mask_adaptive_sided(bgr: np.ndarray, mouth_inner: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    idx = mouth_inner > 0
    if np.count_nonzero(idx) < 80:
        return np.zeros((h, w), np.uint8)

    ys, xs = np.where(idx)
    x_min, x_max = xs.min(), xs.max()
    x_mid = (x_min + x_max) // 2

    mask = np.zeros((h, w), np.uint8)

    for side in ("left", "right"):
        if side == "left":
            side_mask = idx & (np.arange(w)[None, :] <= x_mid)
        else:
            side_mask = idx & (np.arange(w)[None, :] >= x_mid)

        if np.count_nonzero(side_mask) < 30:
            continue

        Ls = L[side_mask].astype(np.float32)
        Bs = B[side_mask].astype(np.float32)

        thr_L = np.percentile(Ls, 58)
        thr_B = np.percentile(Bs, 88)

        cand = (L > thr_L) & (B < thr_B + 6) & side_mask
        mask[cand] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    return mask

# ---------- 3) brutālais ----------
def teeth_mask_brutal(mouth_inner: np.ndarray) -> np.ndarray:
    h, w = mouth_inner.shape[:2]
    ys, xs = np.where(mouth_inner > 0)
    if ys.size == 0:
        return np.zeros_like(mouth_inner)
    y_min, y_max = ys.min(), ys.max()
    mouth_h = y_max - y_min + 1
    band_h = int(mouth_h * 0.45)
    yc = (y_min + y_max) // 2
    y1 = max(y_min, yc - band_h // 2)
    y2 = min(y_max, yc + band_h // 2)
    m = np.zeros_like(mouth_inner)
    m[y1:y2, :] = mouth_inner[y1:y2, :]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.erode(m, k3, iterations=1)
    return m

# ---------- simetrizācija tumšajai pusei ----------
def symmetrize_if_unbalanced(mask: np.ndarray, mouth_inner: np.ndarray) -> np.ndarray:
    """
    Ja viena mutes pusei ir ļoooti maz zobu, bet otrai daudz → pārnesam
    “bagāto” pusi uz “nabaga” pusi ar spoguļošanu.
    """
    h, w = mask.shape[:2]
    ys, xs = np.where(mouth_inner > 0)
    if xs.size == 0:
        return mask
    x_min, x_max = xs.min(), xs.max()
    x_mid = (x_min + x_max) // 2

    left = mask[:, x_min:x_mid+1]
    right = mask[:, x_mid:x_max+1]

    left_count = np.count_nonzero(left)
    right_count = np.count_nonzero(right)

    # ja abas plus mīnus līdzīgas – neko nedaram
    if left_count == 0 and right_count == 0:
        return mask

    imbalance = False
    copy_left_to_right = False
    copy_right_to_left = False

    if left_count < right_count * 0.25:
        imbalance = True
        copy_right_to_left = True
    elif right_count < left_count * 0.25:
        imbalance = True
        copy_left_to_right = True

    if not imbalance:
        return mask

    new_mask = mask.copy()
    if copy_right_to_left and right_count > 0:
        mirrored = np.fliplr(right)
        # ievietojam kreisajā zonā, bet AND ar mouth_inner
        tgt = new_mask[:, x_min:x_min+mirrored.shape[1]]
        mouth_tgt = mouth_inner[:, x_min:x_min+mirrored.shape[1]]
        tgt[(mirrored > 0) & (mouth_tgt > 0)] = 255
        new_mask[:, x_min:x_min+mirrored.shape[1]] = tgt
    elif copy_left_to_right and left_count > 0:
        mirrored = np.fliplr(left)
        tgt = new_mask[:, x_max-mirrored.shape[1]+1:x_max+1]
        mouth_tgt = mouth_inner[:, x_max-mirrored.shape[1]+1:x_max+1]
        tgt[(mirrored > 0) & (mouth_tgt > 0)] = 255
        new_mask[:, x_max-mirrored.shape[1]+1:x_max+1] = tgt

    return new_mask

# ---------- pēdējais sānu “outside” ----------
def allow_small_outside(mask: np.ndarray, mouth_inner: np.ndarray, px: int = 6) -> np.ndarray:
    """
    Ļaujam 5–6 px ārpus mutes, bet tikai tajā zobu joslā.
    Tas paņem pēdējo sānu zobu, kas bieži stāv pašā mutes stūrī.
    """
    if px <= 0:
        return mask
    h, w = mask.shape[:2]
    # atrodam aktīvo zobu joslu
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask
    y_min, y_max = ys.min(), ys.max()
    band = np.zeros_like(mask)
    band[y_min:y_max+1, :] = 1

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (px, 3))
    dil = cv2.dilate(mask, k, iterations=1)

    out = mask.copy()
    # atļaujam tur, kur ir josla, pat ja nav mouth_inner
    extra = (dil > 0) & (band > 0)
    out[extra] = 255
    return out

def build_teeth_mask(bgr: np.ndarray, lips_mask: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mouth_inner = shrink_mask(lips_mask, px=max(1, min(h, w)//300))

    # 1) HSV
    mask1 = teeth_mask_hsv(bgr, mouth_inner)
    mouth_px = np.count_nonzero(mouth_inner)
    m1_px = np.count_nonzero(mask1)

    if mouth_px > 0 and (m1_px >= MIN_PX_OK or m1_px / mouth_px >= MIN_RATIO_OK):
        mask = mask1
    else:
        # 2) adaptīvais pa pusēm
        mask2 = teeth_mask_adaptive_sided(bgr, mouth_inner)
        m2_px = np.count_nonzero(mask2)
        if mouth_px > 0 and (m2_px >= MIN_PX_OK or m2_px / mouth_px >= MIN_RATIO_OK):
            mask = mask2
        else:
            # 3) brutālais
            mask = teeth_mask_brutal(mouth_inner)

    # 4) noturam 2 lielākos
    mask = keep_top_components(mask, n=2)

    # 5) izplešam līdz stūriem
    mask = expand_hv(mask, mouth_inner, hx=20, vy=5)

    # 6) ja viena puse “tukša” – simetrizējam
    mask = symmetrize_if_unbalanced(mask, mouth_inner)

    # 7) atļaujam vēl 5–6 px ārpus mutes, lai paņem pēdējo zobu
    mask = allow_small_outside(mask, mouth_inner, px=6)

    # 8) neliels close
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)

    return mask

def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 14, b_shift: int = 22) -> np.ndarray:
    if np.count_nonzero(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mask = teeth_mask > 0
    Ln = L.astype(np.int16)
    Bn = B.astype(np.int16)

    Ln[mask] = np.clip(Ln[mask] + l_gain, 0, 255)
    Bn[mask] = np.clip(Bn[mask] - b_shift, 0, 255)

    out = cv2.cvtColor(cv2.merge([Ln.astype(np.uint8), A, Bn.astype(np.uint8)]), cv2.COLOR_LAB2BGR)
    return out

# ---------- Endpointi ----------
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

        bgr_for_detect = enhance_for_detection(bgr.copy())

        res = face_mesh.process(cv2.cvtColor(bgr_for_detect, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        lips_mask = lips_mask_from_landmarks(h, w, landmarks)

        teeth_mask = build_teeth_mask(bgr_for_detect, lips_mask)

        out = whiten_only_teeth(bgr, teeth_mask, l_gain=14, b_shift=22)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
