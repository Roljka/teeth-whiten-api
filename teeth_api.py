import os, io, base64, requests
import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Set env var OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def load_image_fix_orientation(file_storage, max_side=1280) -> (np.ndarray, Image.Image):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_to_bgr(img), img

def lips_mask_from_landmarks(h, w, landmarks) -> np.ndarray:
    idx = set()
    for a, b in mp_face_mesh.FACEMESH_LIPS:
        idx.add(a); idx.add(b)
    pts = []
    for i in idx:
        lm = landmarks[i]
        x = int(lm.x * w); y = int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    # mutē mazliet “ieeja” iekšā, lai neskar lūpu robežas
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def build_teeth_mask(bgr: np.ndarray, lips_mask: np.ndarray) -> np.ndarray:
    """ Mutē atlasām potenciālos zobu pikseļus un nedaudz paplašinām,
        lai tumšākas vietas netiktu izlaistas. """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mouth = lips_mask > 0
    candidates = (s < 120) & (v > 90) & mouth  # liberālāk, lai ķertu tumšākus

    mask = np.zeros_like(lips_mask)
    mask[candidates] = 255

    k3 = np.ones((3,3), np.uint8)
    # notīrām trokšņus, tad viegli paplašinām un atkal ieēdam smaganu robežas
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.dilate(mask, k3, iterations=1)
    mask = cv2.erode(mask, k3, iterations=1)

    # drošības slieksnis — nekad ārpus mutes
    mask = cv2.bitwise_and(mask, lips_mask)
    return mask

def png_bytes_from_mask(mask: np.ndarray) -> bytes:
    """ Balts = rediģējamais apgabals. """
    # masku nedaudz “izlīdzinām”, lai nebūtu robi
    mask_smooth = cv2.GaussianBlur(mask, (5,5), 0)
    # jāpārvērš par RGBA ar balto/caurspīdīgo
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0:3] = 255  # balts
    rgba[..., 3] = mask_smooth  # alfa
    ok, buf = cv2.imencode(".png", rgba)
    return buf.tobytes() if ok else None

def jpeg_bytes_from_pil(pil_img: Image.Image, quality=92) -> bytes:
    bio = io.BytesIO()
    pil_img.save(bio, format="JPEG", quality=quality, optimize=True)
    return bio.getvalue()

def call_openai_edit(jpeg_bytes: bytes, mask_png_bytes: bytes, size: str) -> bytes:
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "image": ("input.jpg", jpeg_bytes, "image/jpeg"),
        "mask":  ("mask.png",  mask_png_bytes, "image/png"),
    }
    data = {
        "prompt": ("Whiten only the TEETH to a natural A1–A2 shade. "
                   "Do NOT change lips, gums or skin. Preserve original lighting and textures."),
        "n": 1,
        "size": size,
        "response_format": "b64_json"
    }
    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="Field 'file' missing"), 400

    # 1) ielādējam + EXIF fix + downscale
    bgr, pil_img = load_image_fix_orientation(request.files["file"])
    h, w = bgr.shape[:2]

    # 2) FaceMesh → mutes maska
    res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return jsonify(error="Face not found"), 422

    landmarks = res.multi_face_landmarks[0].landmark
    lips_mask = lips_mask_from_landmarks(h, w, landmarks)

    # 3) zobu maska mutē
    teeth_mask = build_teeth_mask(bgr, lips_mask)

    # drošība — ja maskā 0 pikseļu, nemēģinam editot
    if np.count_nonzero(teeth_mask) < 50:
        # atgriežam oriģinālu (UX ziņā labāk nekā kļūda)
        return send_file(
            io.BytesIO(jpeg_bytes_from_pil(pil_img)),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="whitened.jpg"
        )

    # 4) sagatavojam JPEG un PNG masku
    in_jpeg = jpeg_bytes_from_pil(pil_img)  # izmērs jau ~≤1280px
    mask_png = png_bytes_from_mask(teeth_mask)
    if not mask_png:
        return jsonify(error="Mask encode error"), 500

    # 5) OpenAI edit (izmēru pieskaņojam)
    size = f"{w}x{h}" if max(w, h) <= 1024 else "1024x1024"
    out_bytes = call_openai_edit(in_jpeg, mask_png, size=size)

    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="whitened.jpg"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
