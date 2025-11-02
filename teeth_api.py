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

# -----------------------------------------------------------
# Mediapipe
# -----------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)

# oficiālie mediapipe mutes loki (skat. docs)
MOUTH_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
]
MOUTH_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191
]


# -----------------------------------------------------------
# Palīgfunkcijas
# -----------------------------------------------------------
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
    """Viegls gaišinājums face mesham, bet atpakaļ nesūtām."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


def poly_from_landmarks(h, w, landmarks, indices):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


def mask_from_poly(h, w, pts):
    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask


def build_floor(mask: np.ndarray):
    """atrodam mutes apakšējo malu pa x, lai nogrieztu apakšlūpu"""
    h, w = mask.shape[:2]
    floor = np.full(w, -1, dtype=np.int32)
    ys, xs = np.where(mask > 0)
    for x in range(w):
        col = ys[xs == x]
        if col.size > 0:
            floor[x] = col.max()
    return floor


def cut_below(mask: np.ndarray, floor: np.ndarray, lift_px: int) -> np.ndarray:
    """nogriež zem apakšlūpas (lift_px = cik pikseļus augšup paceļamies)"""
    h, w = mask.shape[:2]
    out = mask.copy()
    for x in range(w):
        y = floor[x]
        if y >= 0:
            out[y + lift_px:h, x] = 0
    return out


def gums_mask(bgr: np.ndarray, mouth_mask: np.ndarray, loose: bool = False) -> np.ndarray:
    """
    Atrod smaganas / lūpu iekšpusi mutē.
    Ja loose=True – mazāk agresīvs (sliktam apgaismojumam).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)

    if not loose:
        red = (((H <= 12) | (H >= 170)) & (S > 35))
        pink = (A > 156)
    else:
        # sliktā gaismā un siltos toņos zobi paši ir siltāki → sliekšņus mīkstinām
        red = (((H <= 10) | (H >= 172)) & (S > 42))
        pink = (A > 165)

    gum = np.zeros_like(mouth_mask)
    gum[(mouth_mask > 0) & (red | pink)] = 255

    gum = cv2.morphologyEx(
        gum,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return gum


# -----------------------------------------------------------
# Zobu maska
# -----------------------------------------------------------
def build_teeth_mask_normal(bgr: np.ndarray, landmarks) -> np.ndarray:
    """mūsu labais, “gaišais” variants – TAS, KO JĀSAGLABĀ"""
    h, w = bgr.shape[:2]

    outer_pts = poly_from_landmarks(h, w, landmarks, MOUTH_OUTER)
    inner_pts = poly_from_landmarks(h, w, landmarks, MOUTH_INNER)

    outer_mask = mask_from_poly(h, w, outer_pts)
    inner_mask = mask_from_poly(h, w, inner_pts)

    # horizontāli paplašinām, lai paņem sānu zobus
    inner_wide = cv2.dilate(
        inner_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (33, 5)),
        iterations=1,
    )

    # lūpa = ārējā - iekšējā
    lips_only = cv2.subtract(outer_mask, cv2.dilate(inner_mask, None, iterations=1))

    # nogriežam apakšlūpu
    floor = build_floor(outer_mask)
    inner_wide = cut_below(inner_wide, floor, lift_px=2)
    lips_only = cut_below(lips_only, floor, lift_px=0)

    # smaganas
    gum = gums_mask(bgr, inner_wide, loose=False)

    # zobu maska
    teeth_mask = cv2.subtract(inner_wide, gum)
    teeth_mask = cv2.subtract(teeth_mask, lips_only)

    # izlīdzinām
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, k3, iterations=1)

    return teeth_mask, inner_wide, lips_only


def build_teeth_mask_lowlight(bgr: np.ndarray, inner_wide: np.ndarray, lips_only: np.ndarray) -> np.ndarray:
    """
    LOW-LIGHT režīms:
    - mazāk paceļam apakšlūpu
    - smaganas tīrām maigāk
    - atstājam vairāk mutes iekšpuses
    """
    h, w = bgr.shape[:2]

    # apakšlūpu šoreiz neaiztiekam gandrīz nemaz
    floor = build_floor(inner_wide)
    inner_ll = cut_below(inner_wide, floor, lift_px=1)

    # maigāks smaganu filtrs
    gum_ll = gums_mask(bgr, inner_ll, loose=True)

    teeth_ll = cv2.subtract(inner_ll, gum_ll)
    teeth_ll = cv2.subtract(teeth_ll, lips_only)

    # ja joprojām par maz – ņemam visu mutes caurumu (bez lūpām)
    filled = np.count_nonzero(teeth_ll)
    mouth_area = np.count_nonzero(inner_ll)
    if mouth_area > 0 and filled / mouth_area < 0.45:
        teeth_ll = cv2.subtract(inner_ll, lips_only)
        teeth_ll = cv2.subtract(teeth_ll, gum_ll)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    teeth_ll = cv2.morphologyEx(teeth_ll, cv2.MORPH_CLOSE, k3, iterations=1)

    return teeth_ll


# -----------------------------------------------------------
# Balināšana
# -----------------------------------------------------------
def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 10, b_shift: int = 22) -> np.ndarray:
    if np.count_nonzero(teeth_mask) == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask = teeth_mask > 0

    Ln = L.astype(np.int16)
    Bn = B.astype(np.int16)
    Ln[mask] = np.clip(Ln[mask] + l_gain, 0, 255)
    Bn[mask] = np.clip(Bn[mask] - b_shift, 0, 255)

    out = cv2.cvtColor(
        cv2.merge([Ln.astype(np.uint8), A, Bn.astype(np.uint8)]),
        cv2.COLOR_LAB2BGR
    )
    return out


# -----------------------------------------------------------
# API
# -----------------------------------------------------------
@app.route("/health")
def health():
    return jsonify(ok=True)


@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="file missing"), 400

        bgr = load_image_fix_orientation(request.files["file"])
        h, w = bgr.shape[:2]

        det_img = enhance_for_detection(bgr)
        res = face_mesh.process(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark

        # 1) palaidam LABO režīmu
        teeth_mask, inner_wide, lips_only = build_teeth_mask_normal(det_img, landmarks)

        # 2) novērtējam – vai nav “pārāk maz” (tipiski tava tumšā bilde)
        filled = np.count_nonzero(teeth_mask)
        mouth_area = np.count_nonzero(inner_wide)

        need_lowlight = False
        if mouth_area == 0:
            need_lowlight = True
        else:
            ratio = filled / mouth_area
            # ja balinām tikai 20-30% mutes → pārslēdzamies
            if ratio < 0.32:
                need_lowlight = True

        if need_lowlight:
            teeth_mask = build_teeth_mask_lowlight(det_img, inner_wide, lips_only)

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
