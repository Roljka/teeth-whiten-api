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

# ---------- Mediapipe FaceMesh (ātrs, statisks, 1 seja) ----------
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

def lips_mask_from_landmarks(h, w, landmarks) -> np.ndarray:
    """
    Veido mutes/lūpu aizpildītu masku no FACEMESH_LIPS punktiem (konveksa čaula).
    """
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
    """Eroze (iekšup) par px, lai netrāpītu lūpu robežām."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1,2*px+1), max(1,2*px+1)))
    return cv2.erode(mask, k, iterations=1)

def build_teeth_mask(bgr: np.ndarray, lips_mask: np.ndarray) -> np.ndarray:
    """
    Zobu maska tikai mutes iekšpusē:
      1) Mute = lūpu maskas erozēta versija (atvirzāmies no lūpām)
      2) HSV nosacījumi: zobi parasti ar zemu S un augstu V
      3) Izgriežam sarkanos toņus (lūpas) pēc H (+ pietiekams S)
      4) Morfoloģija + noturīgāko laukumu atlase (augšējā/apakšējā zoba josla)
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 1) Mutes iekšpuse – atvirzāmies no lūpu robežas
    mouth_inner = shrink_mask(lips_mask, px=max(1, min(h, w)//300))  # adaptīvi 1–3 px

    # 2) Zobu kandidāti: zems S, augsts V, tikai mutes iekšpusē
    #   (pielāgoti sliekšņi, kas labi strādā iekštelpās)
    teeth_cand = (S < 85) & (V > 150) & (mouth_inner > 0)

    # 3) Izgriežam lūpas (sarkans tonis): H ap 0..15 vai 165..180 ar pietiekamu S
    red_like = (((H <= 12) | (H >= 170)) & (S > 30))  # “sarkanie” pikseļi
    teeth_cand = teeth_cand & (~red_like)

    # 4) Morfoloģija: notīram trokšņus un salīmējam zobu laukumus
    mask = np.zeros((h, w), np.uint8)
    mask[teeth_cand] = 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)   # mazus trokšņus ārā
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)  # salīmē zobu gabalus
    mask = cv2.erode(mask, k3, iterations=1)                           # vēl atvirzamies no lūpām
    mask = cv2.dilate(mask, k3, iterations=1)

    # 5) Saglabājam 2 lielākos komponentus (augšējie/apakšējie zobi)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Komponents 0 ir fons
        areas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            areas.append((area, i))
        areas.sort(reverse=True)
        keep = [idx for (_, idx) in areas[:2]]  # top-2
        filt = np.zeros_like(mask)
        for i in keep:
            filt[labels == i] = 255
        mask = filt

    return mask

def whiten_only_teeth(bgr: np.ndarray, teeth_mask: np.ndarray,
                      l_gain: int = 14, b_shift: int = 22) -> np.ndarray:
    """
    Balināšana LAB telpā TIKAI maskā:
      - palielinām L (gaišāks)
      - samazinām b* (mazāk dzeltena)
    """
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

        # Face mesh
        res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        lips_mask = lips_mask_from_landmarks(h, w, landmarks)

        # Zobu maska tikai mutes iekšpusē, ar lūpu izgriešanu
        teeth_mask = build_teeth_mask(bgr, lips_mask)

        # Balinām tikai masku
        out = whiten_only_teeth(bgr, teeth_mask, l_gain=14, b_shift=22)

        # Kodējam JPEG atmiņā
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
