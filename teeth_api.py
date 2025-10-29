# teeth_api.py
import io
import os
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# ---- Tweakable params ----
L_MIN = 60          # LAB L* min  (60–65 parasti labi)
A_MAX = 10          # LAB a* max  (maz reds)
B_MAX = 18          # LAB b* max  (maz yellows)
SAT_MAX = 90        # HSV S max   (zemāka piesāt., lai ignorē lūpu rozā)
WHITEN_BOOST = 22   # cik daudz pacelt L* (10–30)
L_CAP = 92          # balināšanas griesti (90–94 izskatās naturāli)
FEATHER_PX = 6      # maskas mīkstā mala pikseļos

# ---- MediaPipe FaceMesh (lazy init, single instance) ----
_mp = None
_facemesh = None

def get_facemesh():
    global _mp, _facemesh
    if _facemesh is None:
        import mediapipe as mp
        _mp = mp
        _facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,        # dod precīzākas lūpas
            min_detection_confidence=0.5
        )
    return _mp, _facemesh

# Lūpu landmarķu indeksi (MediaPipe FaceMesh)
# Outer lips aptuvenie punkti:
OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
# Inner lips (mutas atvērums) – tieši “zobu logam”
INNER_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]

app = Flask(__name__)
CORS(app)

def landmarks_to_polygon(landmarks, idxs, w, h):
    pts = []
    for i in idxs:
        p = landmarks[i]
        pts.append((int(p.x * w), int(p.y * h)))
    return np.array(pts, dtype=np.int32)

def soft_dilate(mask, radius):
    if radius <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    return cv2.dilate(mask, k, iterations=1)

def feather(mask, px):
    if px <= 0: return mask.astype(np.float32)/255.0
    dist = cv2.distanceTransform((mask==0).astype(np.uint8), cv2.DIST_L2, 3)
    edge = np.clip(dist, 0, px) / float(px)
    # invert edge: 1 iekšā, uz 0 pie malas
    feathered = (1.0 - (1.0 - edge)) * (mask>0)
    # normalizācija
    feathered = np.clip(feathered, 0, 1).astype(np.float32)
    return feathered

def whiten_teeth_bounded(img_bgr):
    mp, facemesh = get_facemesh()

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = facemesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr  # nav seja – atgriežam sākotnējo

    lms = res.multi_face_landmarks[0].landmark

    # Poligoni
    poly_outer = landmarks_to_polygon(lms, OUTER_IDX, w, h)
    poly_inner = landmarks_to_polygon(lms, INNER_IDX, w, h)

    lips_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lips_mask, [poly_outer], 255)
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(inner_mask, [poly_inner], 255)

    # Teeth ROI = inner (mutas atvērums). Lips = outer - inner (neizmantojam).
    teeth_roi = inner_mask.copy()

    # Krāsu telpas
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Heuristiska maska: gaišs (L), nav sarkans (a* zem A_MAX), nav dzeltenīgs (b* zem B_MAX),
    # kā arī zema piesātinātība (S zem SAT_MAX). Plus – tikai teeth_roi iekšpusē.
    base_mask = (
        (L >= L_MIN) &
        (A <= (128 + A_MAX)) &  # LAB a* ap 128 centru
        (B <= (128 + B_MAX)) &
        (S <= SAT_MAX)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(base_mask, teeth_roi)

    # Tīrīšana un “salīmēšana”
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = soft_dilate(mask, 1)

    # Mīksta mala
    alpha = feather(mask, FEATHER_PX)  # [0..1]

    # Balinām tikai L* kanālu, saglabājot krāsu
    Lf = L.astype(np.float32)
    boost = WHITEN_BOOST * alpha
    Lf = np.minimum(Lf + boost, L_CAP)
    L_new = np.clip(Lf, 0, 255).astype(np.uint8)

    lab_new = cv2.merge([L_new, A, B])
    out = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    # Izmixējam tikai maskēto zonu (ja alpha nav binārs)
    alpha3 = np.dstack([alpha, alpha, alpha])
    out_mix = (alpha3 * out + (1 - alpha3) * img_bgr).astype(np.uint8)
    return out_mix

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "Teeth Whitening API – precise teeth-only v3"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    data = f.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Bad image"}), 400

    try:
        out = whiten_teeth_bounded(img)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return jsonify({"error": "Encode failed"}), 500

    return send_file(
        io.BytesIO(enc.tobytes()),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="whitened.jpg",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
