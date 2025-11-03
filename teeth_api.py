# teeth_api.py
# Flask + CORS + /health + /whiten (multipart/form-data: file)
# Saglabā Tavu esošo balināšanas loģiku, ja ir pieejama whiten_teeth(image_bgr, strength)

import io
import os
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)

# CORS: atļaujam visus originus (ja vajag stingrāk – ieliec domēnu sarakstu)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# -----------------------------------------------------------------------------
# Palīgfunkcijas
# -----------------------------------------------------------------------------
def _decode_image_bgr_from_request(file_storage):
    """Nolasa uploadēto failu (werkzeug FileStorage) -> OpenCV BGR attēls."""
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Neizdevās nolasīt attēlu (bojāts vai neatbalstīts formāts).")
    return img

def _jpeg_bytes_from_bgr(img_bgr, quality=92):
    """OpenCV BGR -> JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Neizdevās enkodēt JPEG.")
    return io.BytesIO(buf.tobytes())

# Rezerves “maigs” balinātājs, ja nav tava specializētā funkcija
def _fallback_soft_whiten(bgr, strength: float = 0.7):
    """
    Saudzīgs balināšanas variants: L*a*b* telpā palielina 'L', nedaudz samazina 'b'
    tikai gaišākajos mutes apgabala toņos (vienkāršs krāsu maskējums),
    lai nekļūst par 'krāsojamo grāmatu'.
    """
    img = bgr.copy()

    # Vienkāršs mutes apgabala reģions: ap sejas centru (neideāli, bet droši)
    h, w = img.shape[:2]
    cx, cy = w // 2, int(h * 0.6)
    rx, ry = int(w * 0.28), int(h * 0.18)
    mouth_roi = img[max(cy-ry,0):min(cy+ry,h), max(cx-rx,0):min(cx+rx,w)]

    if mouth_roi.size == 0:
        return img

    lab = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Heuristika “zobu” maskai pēc spilgtuma un ‘b’ (dzeltenā) komponentes
    # (gaišāki + mazāk dzelteni reģioni)
    # Sliekšņi pielāgoti droši; ja vajag agresīvāk, tos var celt/mašināt UI līmenī.
    L_norm = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    B_norm = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
    teeth_mask = cv2.inRange(L_norm, 140, 255) & cv2.inRange(B_norm, 0, 170)

    # Mazgudrā attīrīšana: aizver mazas spraugas, noņem trokšņus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Balinām: pacelam L un mazliet samazinam B (pret dzelteno)
    L_f = L.astype(np.float32)
    B_f = B.astype(np.float32)

    # Spēks (0..1) – piesardzīgs
    s = float(np.clip(strength, 0.0, 1.0))
    L_boost = 18.0 + 40.0 * s
    B_pull  = 6.0 + 12.0 * s

    # pielietojam tikai tur, kur maska
    mask = (teeth_mask > 0)
    L_f[mask] = np.clip(L_f[mask] + L_boost, 0, 255)
    B_f[mask] = np.clip(B_f[mask] - B_pull, 0, 255)

    L2 = L_f.astype(np.uint8)
    B2 = B_f.astype(np.uint8)
    lab2 = cv2.merge([L2, A, B2])
    out_roi = cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)

    out = img.copy()
    out[max(cy-ry,0):min(cy+ry,h), max(cx-rx,0):min(cx+rx,w)] = out_roi
    return out

# Mēģinām importēt Tavu “īstā” balinātāja funkciju, ja projektā tāda jau ir
# (piem., no cita .py faila). Ja nav – izmantosies _fallback_soft_whiten.
_whiten_impl = None
try:
    # from my_whiten_module import whiten_teeth  # piemērs
    # _whiten_impl = whiten_teeth
    pass
except Exception:
    _whiten_impl = None

def whiten_dispatch(image_bgr: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """
    Vienotais ieejs balināšanai: ja ir Tava funkcija, izmanto to; citādi – fallback.
    Tavas funkcijas paraksts sagaidīts: (image_bgr: np.ndarray, strength: float) -> np.ndarray
    """
    if _whiten_impl is not None:
        try:
            return _whiten_impl(image_bgr, strength=strength)
        except Exception:
            # Ja kaut kas noiet greizi – nelaužam API, krītam uz drošu variantu
            return _fallback_soft_whiten(image_bgr, strength=strength)
    else:
        return _fallback_soft_whiten(image_bgr, strength=strength)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/whiten", methods=["POST"])
def whiten():
    """
    Pieņem multipart/form-data ar lauku `file` (image), neobligātu `strength` (0..1).
    Atgriež balināto attēlu kā image/jpeg.
    """
    if "file" not in request.files:
        return jsonify({"error": "missing file field 'file' (multipart/form-data)"}), 400

    file = request.files["file"]
    try:
        img_bgr = _decode_image_bgr_from_request(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # neobligāts stiprums (0..1), default ~0.75
    try:
        strength = float(request.form.get("strength", "0.75"))
    except Exception:
        strength = 0.75
    strength = float(np.clip(strength, 0.0, 1.0))

    try:
        out_bgr = whiten_dispatch(img_bgr, strength=strength)
        buf = _jpeg_bytes_from_bgr(out_bgr, quality=92)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": f"processing failed: {e}"}), 500

# -----------------------------------------------------------------------------
# Lokāla palaišana
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Lokāli: python teeth_api.py
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
