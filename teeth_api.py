import io
import os
import cv2
import base64
import numpy as np
from PIL import Image, ImageOps

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import mediapipe as mp
from openai import OpenAI

# -------------- Flask --------------
app = Flask(__name__)
CORS(app)

# -------------- OpenAI -------------
# Nepieciešams env: OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------- MediaPipe -----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# -------------- Palīgfunkcijas ------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1600) -> (np.ndarray, Image.Image):
    """
    Nolasa bildi, salabo EXIF un (ja vajag) samazina līdz max_side.
    Atgriež (bgr_np, pil_img_resized)
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    bgr = pil_to_bgr(img)
    return bgr, img

def _poly_mask_from_connections(h, w, landmarks, connections) -> np.ndarray:
    idx = set()
    for a, b in connections:
        idx.add(a); idx.add(b)
    pts = []
    for i in idx:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)

    m = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(m, hull, 255)
    return m

def lips_masks_from_landmarks(h, w, landmarks):
    outer = _poly_mask_from_connections(h, w, landmarks, mp_face_mesh.FACEMESH_LIPS)
    inner = cv2.erode(outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=2)
    return outer, inner

def build_teeth_mask(bgr: np.ndarray, inner_mouth_mask: np.ndarray) -> np.ndarray:
    """
    Veido konservatīvu zobu masku mutes iekšpusē
    (S<V slieksņi + adaptīvs uz L, morfo + feather).
    """
    h, w = bgr.shape[:2]
    mouth = inner_mouth_mask > 0

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Ln = clahe.apply(L)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    hsv_cand = ((S < 120) & (V > 110)) & mouth

    Ln_mouth = Ln.copy()
    Ln_mouth[~mouth] = 0
    adap = cv2.adaptiveThreshold(Ln_mouth, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, -5)

    combo = cv2.bitwise_or(adap, (hsv_cand.astype(np.uint8) * 255))
    combo = cv2.bitwise_and(combo, inner_mouth_mask)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(combo, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.erode(mask, k3, iterations=1)

    # feather
    m = (mask.astype(np.float32) / 255.0)
    m = cv2.GaussianBlur(m, (0,0), 6)
    m = np.clip(m, 0, 1)
    mask_soft = (m * 255).astype(np.uint8)
    return mask_soft

def make_inpaint_mask_rgba(pil_img: Image.Image, teeth_mask_u8: np.ndarray) -> Image.Image:
    """
    OpenAI inpaint mask: caurspīdīgās zonas tiks rediģētas.
    Mēs gribam, lai **zobi** ir EDIT zonas => alpha=0 tur, kur zobu maska.
    Citur alpha=255 (saglabāt).
    """
    w, h = pil_img.size
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    # teeth_mask_u8 ir tādā pašā izmērā kā pil_img
    # pārvēršam par alpha: 0 (=edit) uz zobiem, citur 255
    alpha = Image.fromarray((255 - teeth_mask_u8).astype(np.uint8), mode="L")
    mask.putalpha(alpha)
    return mask

def resize_for_openai(pil_img: Image.Image, max_side=1024) -> Image.Image:
    w, h = pil_img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return pil_img

# -------------- Endpoints -----------
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: send multipart/form-data field 'file'."), 400

        # 1) Ielādē + EXIF fix + samazina (CPU/RAM taup.)
        bgr, pil_img = load_image_fix_orientation(request.files["file"], max_side=1600)
        h, w = bgr.shape[:2]

        # 2) FaceMesh -> mutes iekšējā maska
        res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422

        lms = res.multi_face_landmarks[0].landmark
        _, inner_mouth_mask = lips_masks_from_landmarks(h, w, lms)

        # 3) Zobu maska
        teeth_mask = build_teeth_mask(bgr, inner_mouth_mask)
        if int(teeth_mask.max()) == 0:
            return jsonify(error="Teeth not confidently detected"), 422

        # 4) Pielāgo izmēru OpenAI (<=1024) — **attēlu un masku vienādi**
        pil_1024 = resize_for_openai(pil_img, 1024)
        scale_x = pil_1024.size[0] / w
        scale_y = pil_1024.size[1] / h
        teeth_mask_resized = cv2.resize(teeth_mask, (pil_1024.size[0], pil_1024.size[1]), interpolation=cv2.INTER_LINEAR)

        # 5) Izveido RGBA inpaint masku (alpha=0 tikai uz zobiem)
        inpaint_mask = make_inpaint_mask_rgba(pil_1024, teeth_mask_resized)

        # 6) Sagatavo failus atmiņā priekš OpenAI edits
        img_bytes = io.BytesIO()
        pil_1024.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        mask_bytes = io.BytesIO()
        inpaint_mask.save(mask_bytes, format="PNG")
        mask_bytes.seek(0)

        # 7) OpenAI inpaint: “whiten teeth naturally ONLY inside mask”
        prompt = (
            "Whiten the teeth naturally ONLY inside the transparent mask region. "
            "Do not modify lips, gums, skin, lighting, or any other part of the image. "
            "Keep identity and colors unchanged. Subtle, realistic result."
        )

        resp = client.images.edits(
            model="gpt-image-1",
            image=img_bytes,
            mask=mask_bytes,
            prompt=prompt,
            size="1024x1024"
        )

        b64 = resp.data[0].b64_json
        out_bytes = base64.b64decode(b64)

        # 8) Atbilde kā JPEG (lai būtu mazāks)
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        out_buf = io.BytesIO()
        out_img.save(out_buf, format="JPEG", quality=92)
        out_buf.seek(0)

        return send_file(out_buf, mimetype="image/jpeg", download_name="whitened.jpg")

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
