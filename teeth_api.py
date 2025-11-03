import io
import cv2
import numpy as np
from PIL import Image, ExifTags
from flask import Flask, request, send_file, jsonify
import mediapipe as mp

app = Flask(__name__)
mp_face = mp.solutions.face_mesh

# --------------- helpers -----------------

def _fix_orientation(pil_img: Image.Image) -> Image.Image:
    try:
        exif = pil_img._getexif()
        if exif is not None:
            for k, v in ExifTags.TAGS.items():
                if v == "Orientation":
                    orient_key = k
                    break
            o = exif.get(orient_key, None)
            if o == 3:
                pil_img = pil_img.rotate(180, expand=True)
            elif o == 6:
                pil_img = pil_img.rotate(270, expand=True)
            elif o == 8:
                pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        pass
    return pil_img

# MediaPipe inner-lips landmark indeksi (468-modeļiem)
INNER_LIPS = [
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,
    317,14,87,178,88,95  # (komplekti var atšķirties, šis strādā stabili)
]

def lips_mask_from_landmarks(h, w, landmarks):
    pts = []
    for idx in INNER_LIPS:
        lm = landmarks[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # neliels paplašinājums + feather, lai aptvertu zobu apakšas/augšas
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    mask = cv2.GaussianBlur(mask, (21,21), 0)
    return mask

def build_teeth_mask(bgr, mouth_mask):
    # Strādājam tikai mutes zonā
    roi = (mouth_mask > 0)

    # 1) HSV – zobi: gaiši, nepiesātināti
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m_hsv = (V > 120) & (S < 100)

    # 2) Lab – izmetam “rozā” (a* liels) un “dzeltens” (b* liels)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    m_lab = (a < 135) & (b < 140) & (L > 120)   # 128 ir neitrāls; zem/virs pielāgojam

    # 3) “invert-green” – invertē, pēc tam izmetam zaļganās vietas (parasti lūpas/smaganas)
    inv = 255 - bgr
    inv_hsv = cv2.cvtColor(inv, cv2.COLOR_BGR2HSV)
    _, invS, invV = cv2.split(inv_hsv)
    # Zaļganās vietas invertētajā attēlā ir ar augstāku S un vidēju V;
    # mēs tās IZMETAM, tātad ņemam pretēju masku:
    not_greenish = ~((invS > 60) & (invV > 60))

    # Kombinācija (tikai mutes iekšienē)
    raw = m_hsv & m_lab & not_greenish & roi

    # Tīrīšana: aizveram spraugas, aizpildām plaknes
    raw_u8 = np.where(raw, 255, 0).astype(np.uint8)
    raw_u8 = cv2.morphologyEx(raw_u8, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 2)

    # Mazie laukumi ārā:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_u8, connectivity=8)
    cleaned = np.zeros_like(raw_u8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 200:  # slieksni var pacelt/mažināt
            cleaned[labels == i] = 255

    # Feather maska, lai malas ir mīkstas
    cleaned = cv2.GaussianBlur(cleaned, (15,15), 0)
    return cleaned

def whiten_teeth(bgr, mask, strength=5):
    """Strength: 1..8"""
    strength = max(1, min(8, int(strength)))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a, b = cv2.split(lab)

    # paceļam L (gaišumu) pret 95, pēc “smoothstep” līknes
    target_L = 240.0  # ~ 95/100 skalā (OpenCV Lab ir 0..255)
    alpha = 0.06 * strength  # koeficients balināšanai
    L_new = L + alpha * (target_L - L)

    # samazinām dzeltenumu (b)
    beta = 0.09 * strength
    b_new = b - beta * np.maximum(0, b - 128)

    # iekš maskas
    m = (mask.astype(np.float32)/255.0)[..., None]
    L = L*(1-m) + L_new*m
    b = b*(1-m) + b_new*m

    out = cv2.merge([np.clip(L,0,255), a, np.clip(b,0,255)]).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_Lab2BGR)
    # neliels “tone-mapping”, lai neizskatās krītiņbalts:
    out_bgr = cv2.detailEnhance(out_bgr, sigma_s=5, sigma_r=0.15)
    return out_bgr

# --------------- API -----------------

@app.route("/whiten", methods=["POST"])
def whiten_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "missing file field 'file' (multipart/form-data)"}), 400

    f = request.files["file"]
    pil = Image.open(f.stream).convert("RGB")
    pil = _fix_orientation(pil)

    # uz OpenCV
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # MediaPipe sejas mesh
    with mp_face.FaceMesh(static_image_mode=True,
                          refine_landmarks=True,
                          max_num_faces=1,
                          min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        # ja neredz seju – sākumā mēģinam bez balināšanas
        out = img.copy()
    else:
        h, w = img.shape[:2]
        lms = res.multi_face_landmarks[0].landmark
        mouth_mask = lips_mask_from_landmarks(h, w, lms)
        teeth_mask = build_teeth_mask(img, mouth_mask)

        # “8 līmeņu” balināšanas skala: var padot ?strength=1..8 (default 5)
        strength = int(request.args.get("strength", "5"))
        out = whiten_teeth(img, teeth_mask, strength=strength)

    # atpakaļ uz JPEG
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")
