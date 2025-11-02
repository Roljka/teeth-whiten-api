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
# Mediapipe – 1 seja, statisks, ātrs
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)

# oficiālā mutes mala no mediapipe
MOUTH_LANDMARKS = set()
for a, b in mp_face_mesh.FACEMESH_LIPS:
    MOUTH_LANDMARKS.add(a)
    MOUTH_LANDMARKS.add(b)


# ============================================================
# Palīgfunkcijas
# ============================================================
def load_image_fix_orientation(file_storage, max_side=1600) -> np.ndarray:
    """Nolasa bildi, izgriež EXIF rotāciju, samazina, atdod BGR."""
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def enhance_for_detection(bgr: np.ndarray) -> np.ndarray:
    """Viegls gaišinājums, lai face mesh labāk redz."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


def mouth_mask_from_landmarks(h: int, w: int, landmarks) -> np.ndarray:
    """Uzzīmē mutes konveksu čaulu no FACEMESH_LIPS punktiem."""
    pts = []
    for idx in MOUTH_LANDMARKS:
        lm = landmarks[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask


def build_safe_floor(mask: np.ndarray) -> np.ndarray:
    """Katram X atrodam mutes apakšu un paceļamies par pāris px augstāk (lai nebalina apakšlūpu)."""
    h, w = mask.shape[:2]
    floor = np.full(w, -1, dtype=np.int32)
    ys, xs = np.where(mask > 0)
    for x in range(w):
        col = ys[xs == x]
        if col.size > 0:
            floor[x] = col.max()
    # paceļam mazliet – 2..4px atkarībā no augstuma
    mouth_h = (ys.max() - ys.min()) + 1 if ys.size else 0
    lift_center = max(2, mouth_h // 20)   # ~2–4px
    return floor, lift_center


def cut_below_floor(mask: np.ndarray, floor: np.ndarray, lift: int) -> np.ndarray:
    """No maskas nogriež visu, kas ir zem apakšlūpas."""
    h, w = mask.shape[:2]
    out = mask.copy()
    for x in range(w):
        y = floor[x]
        if y >= 0:
            out[y + lift:h, x] = 0
    return out


def color_reject_lips_and_gums(bgr: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
    """
    Uztaisa masku ar lūpām/smaganām (rozā/sarkanie), lai vēlāk to varētu atņemt.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)

    red_like = (((H <= 12) | (H >= 170)) & (S > 25))
    pink_like = (A > 156)

    gum = np.zeros_like(mouth_mask)
    gum[(mouth_mask > 0) & (red_like | pink_like)] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gum = cv2.morphologyEx(gum, cv2.MORPH_OPEN, k, iterations=1)
    return gum


# ============================================================
# Te ir mūsu "vienkāršā" zobu maskas loģika
# ============================================================
def build_teeth_mask(bgr: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
    """
    1) izveidojam mutes iekšpusi
    2) mēģinām pēc krāsas atrast zobus
    3) ja par maz – balinām visu mutes iekšpusi (bez lūpas)
    """
    h, w = bgr.shape[:2]

    # mutes iekšpuse – drusku iekšā, lai nenobrauc lūpu robežu
    mouth_inner = cv2.erode(mouth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)

    # apakšlūpas nogriešana
    floor, lift = build_safe_floor(mouth_mask)
    mouth_inner = cut_below_floor(mouth_inner, floor, lift)

    # krāsu filtrs lūpām
    gums = color_reject_lips_and_gums(bgr, mouth_inner)
    mouth_no_gum = cv2.bitwise_and(mouth_inner, cv2.bitwise_not(gums))

    # mēģinām "smuki" pa krāsu
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # tikai mutē
    idx = mouth_no_gum > 0
    if np.count_nonzero(idx) == 0:
        return mouth_inner  # vismaz kaut kas

    Lm = L[idx].astype(np.float32)
    Bm = B[idx].astype(np.float32)

    L_med = np.median(Lm)
    B_med = np.median(Bm)

    # zobs parasti ir gaišāks un mazāk dzeltens par vidējo mutes iekšpusi
    tooth_cand = (L.astype(np.float32) > (L_med - 5)) & (B.astype(np.float32) < (B_med + 18)) & idx

    teeth_mask = np.zeros((h, w), np.uint8)
    teeth_mask[tooth_cand] = 255

    # salīmējam zobus, paplašinām horizontāli (lai paņem sānu)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, k3, iterations=1)
    teeth_mask = cv2.dilate(teeth_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)), 1)

    # ja tomēr rezultāts ir par maz → fallback: balini visu muti (bez lūpas)
    filled_ratio = np.count_nonzero(teeth_mask) / float(np.count_nonzero(mouth_inner)) if np.count_nonzero(mouth_inner) else 0
    if filled_ratio < 0.25:
        teeth_mask = mouth_no_gum.copy()
        # mazliet izlīdzinām
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, k3, iterations=1)

    return teeth_mask


# ============================================================
# Balināšana – tikai maskā
# ============================================================
def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 10, b_shift: int = 24) -> np.ndarray:
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


# ============================================================
# API
# ============================================================
@app.route("/health")
def health():
    return jsonify(ok=True)


@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="file is missing"), 400

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        # face mesh
        for_mesh = enhance_for_detection(bgr)
        res = face_mesh.process(cv2.cvtColor(for_mesh, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        mouth_mask = mouth_mask_from_landmarks(h, w, landmarks)

        teeth_mask = build_teeth_mask(for_mesh, mouth_mask)

        out = whiten_only_teeth(bgr, teeth_mask)

        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify(error="encode failed"), 500

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
