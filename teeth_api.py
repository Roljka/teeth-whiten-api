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

# ---------- Mediapipe FaceMesh ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# cik daudz zobu pikseļu uzskatām par "ok" (ja zem šī – slēdzam adaptīvo režīmu)
MIN_TEETH_PX_RATIO = 0.22   # 22% no mutes iekšpuses
MIN_TEETH_PX_ABS = 350      # vai vismaz tik pikseļus

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

def enhance_for_detection(bgr: np.ndarray) -> np.ndarray:
    """Maigs izgaismojums tikai detekcijai."""
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, 2*px+1), max(1, 2*px+1)))
    return cv2.erode(mask, k, iterations=1)

def expand_teeth_sideways(teeth_mask: np.ndarray, mouth_inner: np.ndarray, px: int = 11) -> np.ndarray:
    """izplešam horizontāli, bet tikai mutes iekšpusē"""
    if px <= 0:
        return teeth_mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (px, 3))
    dil = cv2.dilate(teeth_mask, k, iterations=1)
    out = np.zeros_like(teeth_mask)
    out[(dil > 0) & (mouth_inner > 0)] = 255
    return out

def keep_top_components(mask: np.ndarray, max_components: int = 2) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        areas.append((area, i))
    areas.sort(reverse=True)
    keep = [idx for (_, idx) in areas[:max_components]]
    filt = np.zeros_like(mask)
    for i in keep:
        filt[labels == i] = 255
    return filt

def hsv_teeth_mask(bgr: np.ndarray, mouth_inner: np.ndarray) -> np.ndarray:
    """
    Ātrais variants – strādā labā gaismā.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # nedaudz pielaidīgs
    cand = (S < 105) & (V > 135) & (mouth_inner > 0)

    # izmetam sarkanos
    red_like = (((H <= 12) | (H >= 170)) & (S > 30))
    cand = cand & (~red_like)

    mask = np.zeros((h, w), np.uint8)
    mask[cand] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)

    return mask

def adaptive_teeth_mask(bgr: np.ndarray, mouth_inner: np.ndarray) -> np.ndarray:
    """
    Fallback sliktai gaismai:
      - skatāmies tikai mutes iekšienē
      - paņemam gaišāko daļu no tās (percentile)
      - atmetam sarkanos
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mouth_idx = mouth_inner > 0
    if np.count_nonzero(mouth_idx) < 50:
        # nav mutes?
        return np.zeros((h, w), np.uint8)

    Lm = L[mouth_idx].astype(np.float32)
    Bm = B[mouth_idx].astype(np.float32)

    # slieksni izvēlamies relatīvi – augšējie 40% pēc gaišuma
    thr_L = np.percentile(Lm, 60)  # ja tumša bilde – šis būs zemāks
    # izmetam dzeltenīgo / siltu (smaganas) – pēc b kanāla
    thr_B = np.percentile(Bm, 85)  # siltākajiem (smaganām) b būs lielāks

    cand = (L > thr_L) & (B < thr_B + 5) & mouth_idx

    mask = np.zeros((h, w), np.uint8)
    mask[cand] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    return mask

def build_teeth_mask(bgr: np.ndarray, lips_mask: np.ndarray) -> np.ndarray:
    """
    Galvenā maskas būvēšana:
      1) mutes iekšpuse
      2) mēģinām HSV
      3) ja HSV par maz – adaptīvais režīms
      4) saglabājam lielākos gabalus
      5) izplešam horizontāli, lai paņem arī sānu zobus
    """
    h, w = bgr.shape[:2]
    mouth_inner = shrink_mask(lips_mask, px=max(1, min(h, w)//300))

    # 1) Mēģinām HSV
    mask_hsv = hsv_teeth_mask(bgr, mouth_inner)

    mouth_px = np.count_nonzero(mouth_inner)
    hsv_px = np.count_nonzero(mask_hsv)

    need_adaptive = False
    if hsv_px < MIN_TEETH_PX_ABS:
        need_adaptive = True
    elif mouth_px > 0 and (hsv_px / float(mouth_px)) < MIN_TEETH_PX_RATIO:
        need_adaptive = True

    if not need_adaptive:
        mask = mask_hsv
    else:
        # 2) adaptīvais – sliktai gaismai
        mask = adaptive_teeth_mask(bgr, mouth_inner)

    # 3) saglabājam augšējos/apakšējos
    mask = keep_top_components(mask, max_components=2)

    # 4) izplešam horizontāli, lai paņem sānos
    mask = expand_teeth_sideways(mask, mouth_inner, px=11)

    # 5) neliels close, lai nav caurumi
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

        # gaišā kopija tikai detekcijai
        bgr_for_detect = enhance_for_detection(bgr.copy())

        # face mesh
        res = face_mesh.process(cv2.cvtColor(bgr_for_detect, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        landmarks = res.multi_face_landmarks[0].landmark
        lips_mask = lips_mask_from_landmarks(h, w, landmarks)

        # zobu maska ar adaptīvo fallback
        teeth_mask = build_teeth_mask(bgr_for_detect, lips_mask)

        # balinām tikai masku, uz oriģināla
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
