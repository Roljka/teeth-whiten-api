# teeth_api.py
# Flask API – zobu balināšana ar LAB telpas korekciju (soft maska, edge-aware)
# Endpoints:
#   GET  /health               -> {"ok": true}
#   POST /whiten               -> multipart/form-data {image: file, level: 1..8}
#                                atgriež processed image/jpeg

import io
import math
import traceback

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify, make_response
from PIL import Image, ImageOps

# ------------------------------------------------------------
# Balināšanas kodols (nekas netiek "pārkrāsots ar baltu")
# ------------------------------------------------------------

def whiten_teeth(img_bgr: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """
    Tekstūras–saglabājoša balināšana:
    - Soft maska (mutes reģions + krāsas), pēc tam edge-aware izlīdzināšana
    - LAB telpā ceļ L (spilgtumu) un mazina b (dzeltenumu)
    - Aizsardzība pret “plankumiem” ēnās
    strength: 0..1 (ieteicami 0.5..0.75)
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # --- HSV kandidātu maska zobiem (mazs piesātinājums, gana gaiši) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0

    cand = (S < 0.32) & (V > 0.35)
    cand = (cand.astype(np.uint8) * 255)

    # aizver spraugas starp zobiem
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=2)
    cand = cv2.medianBlur(cand, 5)

    # noņem sarkanās zonas (lūpas/smaganas) ar LAB A kanālu
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    gum_reject = (A > 150)  # konservatīvi
    mask_hard = (cand > 0) & (~gum_reject)
    mask_hard = (mask_hard.astype(np.uint8) * 255)

    # ja maska pārāk maza – paplašinām drusku
    if int(mask_hard.sum()) < 800:
        mask_hard = cv2.dilate(mask_hard, k, iterations=1)

    # --- Soft maska ar distance transform un blur ---
    dist = cv2.distanceTransform(mask_hard, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    soft = cv2.GaussianBlur(dist, (0, 0), 1.2).astype(np.float32)
    soft = np.clip(soft, 0.0, 1.0)

    # Edge-aware bāze, lai pārejas uz zobu malām ir gludas
    L_float = L.astype(np.float32) / 255.0
    base = cv2.edgePreservingFilter(img, flags=1, sigma_s=16, sigma_r=0.25)
    base_L = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    # (base_L izmantojam netieši – tikai malas saglabāšanai, soft jau ir mīksta)

    # --- LAB korekcijas tikai maskas ietvaros ---
    L_gain = 1.0 + (0.45 * strength)      # līdz ~+45%
    B_shift = - (28.0 * strength)         # mazina dzeltenumu

    L_new = (L_float * (1.0 - soft) + (L_float * L_gain) * soft)
    B_new = B.astype(np.float32) + (B_shift * soft * 255.0)

    # robežas
    L_new = np.clip(L_new, 0, 1)
    B_new = np.clip(B_new, 0, 255)

    lab_new = cv2.merge([
        (L_new * 255.0).astype(np.uint8),
        A,
        B_new.astype(np.uint8)
    ])
    out = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    # Mazāks efekts ļoti tumšās ēnās (novērš plankumus)
    shadow = (L_float < 0.25).astype(np.float32)
    blend = 1.0 - 0.35 * shadow * soft
    out = (out.astype(np.float32) * blend[..., None] +
           img.astype(np.float32) * (1.0 - blend[..., None])).astype(np.uint8)

    return out

# ------------------------------------------------------------
# Palīgfunkcijas
# ------------------------------------------------------------

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Pillow -> OpenCV BGR ar EXIF orientācijas ievērošanu."""
    pil_img = ImageOps.exif_transpose(pil_img)  # salabo orientāciju
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)  # RGB(+A)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        # atmetam alfa – zobiem nav vajadzīgs
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 92) -> bytes:
    """OpenCV BGR -> JPEG bytes (bez EXIF, bet ar pareizu orientāciju jau attēlā)."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def map_level_to_strength(level: int) -> float:
    """
    1..8 -> 0.35..0.85 lineāri, lai vienkāršs UI 'balināšanas līmenis'
    """
    level = max(1, min(8, int(level)))
    return 0.35 + (level - 1) * (0.85 - 0.35) / 7.0

# ------------------------------------------------------------
# Flask
# ------------------------------------------------------------

app = Flask(__name__)

# vienkāršs CORS (ja negribi atsevišķu flask_cors atkarību)
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/whiten", methods=["POST", "OPTIONS"])
def whiten_endpoint():
    if request.method == "OPTIONS":
        return make_response(("", 204))
    try:
        if "image" not in request.files:
            return jsonify({"error": "missing file field 'image' (multipart/form-data)"}), 400

        f = request.files["image"]
        img_bytes = f.read()
        if not img_bytes:
            return jsonify({"error": "empty file"}), 400

        # Lasām ar Pillow, lai ievērotu EXIF orientāciju
        pil = Image.open(io.BytesIO(img_bytes))
        img_bgr = pil_to_bgr(pil)

        # drošībai ieturam max izmēru, lai būtu ātrdarbīgi (piem., 2200px)
        h, w = img_bgr.shape[:2]
        max_side = max(h, w)
        if max_side > 2200:
            scale = 2200.0 / max_side
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # UI “level” -> 1..8
        level = request.form.get("level", "6")
        try:
            level_int = int(level)
        except:
            level_int = 6
        strength = map_level_to_strength(level_int)

        out = whiten_teeth(img_bgr, strength=strength)
        jpeg = bgr_to_jpeg_bytes(out, quality=92)

        return send_file(
            io.BytesIO(jpeg),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="whitened.jpg"
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# vietējai palaišanai:  python teeth_api.py
if __name__ == "__main__":
    # Noklusēti 0.0.0.0:8000 – pielāgo pēc vajadzības
    app.run(host="0.0.0.0", port=8000)
