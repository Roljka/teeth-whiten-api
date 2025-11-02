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

# ============================================================
# Mediapipe FaceMesh – vieglā, stabilā versija
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# ============================================================
# Oficiālie mutes punkti
# ============================================================
MOUTH_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185, 61
]
MOUTH_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191, 78
]

MIN_RATIO_OK = 0.30
MIN_PX_OK = 350


# ============================================================
# Palīgfunkcijas
# ============================================================
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
    # viegla gamma + CLAHE
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


def lips_mask_from_landmarks_strict(h, w, landmarks) -> np.ndarray:
    outer_pts = []
    for idx in MOUTH_OUTER:
        lm = landmarks[idx]
        outer_pts.append([int(lm.x * w), int(lm.y * h)])
    outer_pts = np.array(outer_pts, dtype=np.int32)

    inner_pts = []
    for idx in MOUTH_INNER:
        lm = landmarks[idx]
        inner_pts.append([int(lm.x * w), int(lm.y * h)])
    inner_pts = np.array(inner_pts, dtype=np.int32)

    mask = np.zeros((h, w), np.uint8)
    if outer_pts.shape[0] >= 3:
        cv2.fillPoly(mask, [outer_pts], 255)
    if inner_pts.shape[0] >= 3:
        cv2.fillPoly(mask, [inner_pts], 0)
    return mask


def shrink_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k, iterations=1)


def get_mouth_metrics(lips_mask: np.ndarray):
    ys, xs = np.where(lips_mask > 0)
    if ys.size == 0:
        return None
    return {
        "x_min": xs.min(),
        "x_max": xs.max(),
        "y_min": ys.min(),
        "y_max": ys.max(),
        "mouth_h": ys.max() - ys.min() + 1,
    }


def lip_floor_curve(lips_mask: np.ndarray) -> np.ndarray:
    h, w = lips_mask.shape[:2]
    floor = np.full(w, -1, dtype=np.int32)
    ys, xs = np.where(lips_mask > 0)
    for x in range(w):
        col = ys[xs == x]
        if col.size > 0:
            floor[x] = col.max()
    return floor


def build_safe_floor(floor: np.ndarray, metrics: dict,
                     center_factor: float = 0.07,
                     side_factor: float = 0.18) -> np.ndarray:
    # mazliet MAZĀKS atvirzījums, lai neapgrieztu apakšējos priekšzobus
    w = floor.shape[0]
    safe = np.full_like(floor, -1)
    x_min = metrics["x_min"]; x_max = metrics["x_max"]
    mouth_h = metrics["mouth_h"]

    off_center = max(1, int(mouth_h * center_factor))
    off_side = max(3, int(mouth_h * side_factor))

    width = x_max - x_min + 1
    c1 = x_min + width // 4
    c2 = x_max - width // 4

    for x in range(x_min, x_max + 1):
        base = floor[x]
        if base < 0:
            continue
        safe[x] = base - (off_center if c1 <= x <= c2 else off_side)
    return safe


def apply_safe_floor(mask: np.ndarray, safe_floor: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    for x in range(w):
        y_cut = safe_floor[x]
        if y_cut >= 0:
            out[y_cut + 1:h, x] = 0
    return out


def light_gum_mask(bgr: np.ndarray, mouth_inner: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    red_like = (((H <= 12) | (H >= 170)) & (S > 30))
    pink = (A > 154)

    gum = np.zeros_like(mouth_inner)
    gum[(mouth_inner > 0) & (red_like | pink)] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gum = cv2.morphologyEx(gum, cv2.MORPH_OPEN, k, iterations=1)
    return gum


# ============================================================
# 1) esošais “smukais” HSV variants
# ============================================================
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


# ============================================================
# 2) adaptīvais LAB variants pa kreisi/pa labi
# ============================================================
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
        side_mask = idx & ((np.arange(w)[None, :] <= x_mid) if side == "left" else (np.arange(w)[None, :] >= x_mid))
        if np.count_nonzero(side_mask) < 30:
            continue

        Ls = L[side_mask].astype(np.float32)
        Bs = B[side_mask].astype(np.float32)
        thr_L = np.percentile(Ls, 56)
        thr_B = np.percentile(Bs, 90)

        cand = (L > thr_L) & (B < thr_B + 8) & side_mask
        mask[cand] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    return mask


# ============================================================
# 3) brutālais – mutes vidējā josla
# ============================================================
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


def keep_top_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    areas.sort(reverse=True)
    keep = [idx for (_, idx) in areas[:n]]
    out = np.zeros_like(mask)
    for i in keep:
        out[labels == i] = 255
    return out


# ============================================================
# 4) KRĀSU SPĒLES (mīkstāks variants, lai nebalina lūpu)
# ============================================================
def make_tooth_color_maps(bgr: np.ndarray, mouth_mask: np.ndarray):
    h, w = bgr.shape[:2]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_norm = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    B_inv = 255 - B
    lab_tooth = cv2.addWeighted(L_norm, 0.65, B_inv, 0.35, 0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    S_inv = 255 - S
    hsv_tooth = cv2.addWeighted(S_inv, 0.45, V, 0.55, 0)

    red_like = (((H <= 12) | (H >= 170)) & (S > 30))
    pink_like = (A > 158)

    delip = np.zeros((h, w), np.uint8)
    delip[red_like] = 255
    delip[pink_like > 0] = 255

    lab_tooth[mouth_mask == 0] = 0
    hsv_tooth[mouth_mask == 0] = 0
    delip[mouth_mask == 0] = 0

    return {
        "lab_tooth": lab_tooth,
        "hsv_tooth": hsv_tooth,
        "delip": delip,
        "A": A,
        "H": H,
        "S": S
    }


def final_anti_lip_filter(bgr: np.ndarray, mouth_mask: np.ndarray, cand_mask: np.ndarray) -> np.ndarray:
    """ņemam to, ko krāsu triks atdeva, un izmetam visu, kas izskatās pēc lūpas/smaganas."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    out = cand_mask.copy()

    red_like = (((H <= 12) | (H >= 170)) & (S > 25))
    pink_like = (A > 158)

    out[(red_like | pink_like) & (mouth_mask > 0)] = 0

    # vēl viens nosacījums – ja ir tumšs (V < 60), tad nav zobs
    out[(V < 55) & (mouth_mask > 0)] = 0

    return out


def build_teeth_mask_from_colors(bgr: np.ndarray,
                                 mouth_mask: np.ndarray,
                                 safe_floor: np.ndarray = None) -> np.ndarray:
    h, w = bgr.shape[:2]
    maps = make_tooth_color_maps(bgr, mouth_mask)
    lab_map = maps["lab_tooth"]
    hsv_map = maps["hsv_tooth"]

    # MĪKSTĀK: nevis AND, bet OR → lai tumšiem zobiem tomēr ir šanse
    lab_bin = (lab_map > 120).astype(np.uint8) * 255
    hsv_bin = (hsv_map > 120).astype(np.uint8) * 255

    comb = cv2.bitwise_or(lab_bin, hsv_bin)

    # nedaudz salīmējam
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, k3, iterations=1)
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k3, iterations=2)

    # horizontāli paplašinām
    comb = cv2.dilate(comb, cv2.getStructuringElement(cv2.MORPH_RECT, (28, 3)), iterations=1)

    # nogriežam zem lūpas
    if safe_floor is not None:
        for x in range(w):
            y_cut = safe_floor[x]
            if y_cut is not None and y_cut >= 0:
                comb[y_cut + 1:h, x] = 0

    # UZLIEKAM anti-lip filtru pašās beigās
    comb = final_anti_lip_filter(bgr, mouth_mask, comb)

    return comb


# ============================================================
# LOW-LIGHT fallback
# ============================================================
def build_teeth_mask_lowlight(bgr: np.ndarray,
                              mouth_inner: np.ndarray,
                              safe_floor: np.ndarray,
                              metrics: dict) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = mouth_inner.copy()

    for x in range(w):
        y_cut = safe_floor[x]
        if y_cut >= 0:
            mask[y_cut + 1:h, x] = 0

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (36, 3)), iterations=1)

    gum = light_gum_mask(bgr, mouth_inner)
    mask[gum > 0] = 0

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)

    # vēlreiz anti-lip – jo tieši te biežāk noplūst
    mask = final_anti_lip_filter(bgr, mouth_inner, mask)

    return mask


# ============================================================
# GALVENĀ – kas saliek VISUS trikus kopā
# ============================================================
def build_teeth_mask(bgr: np.ndarray, lips_mask: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    metrics = get_mouth_metrics(lips_mask)
    if metrics is None:
        return np.zeros((h, w), np.uint8)

    mouth_inner = shrink_mask(lips_mask, px=max(1, min(h, w) // 300))

    lip_floor = lip_floor_curve(lips_mask)
    safe_floor_main = build_safe_floor(lip_floor, metrics,
                                       center_factor=0.07,
                                       side_factor=0.18)

    # 1) HSV
    mask1 = teeth_mask_hsv(bgr, mouth_inner)
    mouth_px = np.count_nonzero(mouth_inner)
    m1_px = np.count_nonzero(mask1)

    if mouth_px > 0 and (m1_px >= MIN_PX_OK or m1_px / mouth_px >= MIN_RATIO_OK):
        base_mask = mask1
    else:
        # 2) adaptīvais
        mask2 = teeth_mask_adaptive_sided(bgr, mouth_inner)
        m2_px = np.count_nonzero(mask2)
        if mouth_px > 0 and (m2_px >= MIN_PX_OK or m2_px / mouth_px >= MIN_RATIO_OK):
            base_mask = mask2
        else:
            # 3) KRĀSU SPĒLES
            base_mask = build_teeth_mask_from_colors(bgr, mouth_inner, safe_floor_main)

            if np.count_nonzero(base_mask) < 120:
                # ja vēl par maz
                base_mask = teeth_mask_brutal(mouth_inner)

    base_mask = keep_top_components(base_mask, n=2)
    base_mask = apply_safe_floor(base_mask, safe_floor_main)

    filled_ratio = np.count_nonzero(base_mask) / float(mouth_px) if mouth_px > 0 else 0.0
    if filled_ratio < 0.25:
        safe_floor_low = build_safe_floor(lip_floor, metrics,
                                          center_factor=0.05,
                                          side_factor=0.15)
        base_mask = build_teeth_mask_lowlight(bgr, mouth_inner, safe_floor_low, metrics)

    return base_mask


# ============================================================
# Balināšana
# ============================================================
def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      base_l_gain: int = 10, max_b_shift: int = 26) -> np.ndarray:
    if np.count_nonzero(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask = teeth_mask > 0

    B_in = B[mask].astype(np.float32)
    if B_in.size == 0:
        return bgr

    b_low = np.percentile(B_in, 35)
    b_high = np.percentile(B_in, 90)
    denom = max(1.0, (b_high - b_low))
    weight = np.clip((B.astype(np.float32) - b_low) / denom, 0, 1)

    Ln = L.astype(np.int16)
    Bn = B.astype(np.int16)

    Ln[mask] = np.clip(Ln[mask] + base_l_gain, 0, 255)
    Bn[mask] = np.clip(Bn[mask] - (weight[mask] * max_b_shift).astype(np.int16), 0, 255)

    out = cv2.cvtColor(cv2.merge([Ln.astype(np.uint8), A, Bn.astype(np.uint8)]), cv2.COLOR_LAB2BGR)
    return out


# ============================================================
# Endpointi
# ============================================================
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

        lips_mask = lips_mask_from_landmarks_strict(h, w, landmarks)

        teeth_mask = build_teeth_mask(bgr_for_detect, lips_mask)

        out = whiten_only_teeth(bgr, teeth_mask,
                                base_l_gain=10,
                                max_b_shift=26)

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
