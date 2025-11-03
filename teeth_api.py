# teeth_api.py
import io, os, base64, re
import cv2, numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- helpers ----------------
def _decode_image_bgr_from_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Neizdevās nolasīt attēlu (bojāts vai neatbalstīts formāts).")
    return img

def _decode_image_bgr_from_filestorage(fs):
    return _decode_image_bgr_from_bytes(fs.read())

def _jpeg_bytes_from_bgr(img_bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Neizdevās enkodēt JPEG.")
    return io.BytesIO(buf.tobytes())

# drošs rezerves balinātājs
def _fallback_soft_whiten(bgr, strength: float = 0.75):
    img = bgr.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, int(h * 0.60)
    rx, ry = int(w * 0.28), int(h * 0.18)
    y1, y2 = max(cy-ry,0), min(cy+ry,h)
    x1, x2 = max(cx-rx,0), min(cx+rx,w)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Ln = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    Bn = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.inRange(Ln, 140, 255) & cv2.inRange(Bn, 0, 170)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    s = float(np.clip(strength, 0.0, 1.0))
    Lf, Bf = L.astype(np.float32), B.astype(np.float32)
    Lf[mask>0] = np.clip(Lf[mask>0] + (18+40*s), 0, 255)
    Bf[mask>0] = np.clip(Bf[mask>0] - (6+12*s), 0, 255)
    out_roi = cv2.cvtColor(cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)]), cv2.COLOR_Lab2BGR)
    out = img.copy()
    out[y1:y2, x1:x2] = out_roi
    return out

# ja Tev ir savs īstais balinātājs, ieliec importu šeit un piešķir _whiten_impl
_whiten_impl = None
def whiten_dispatch(image_bgr, strength=0.75):
    if _whiten_impl is not None:
        try:
            return _whiten_impl(image_bgr, strength=strength)
        except Exception:
            return _fallback_soft_whiten(image_bgr, strength)
    return _fallback_soft_whiten(image_bgr, strength)

def _pick_image_from_request():
    """
    Pieņem:
      - multipart files: 'file' vai 'image' vai pirmais elements request.files
      - form dataURL laukā 'image' (data:image/jpeg;base64,...)
      - raw binary POST ar Content-Type: image/*
    """
    # 1) multipart: konkrēti lauki
    if "file" in request.files:
        return _decode_image_bgr_from_filestorage(request.files["file"])
    if "image" in request.files:
        return _decode_image_bgr_from_filestorage(request.files["image"])
    # 2) multipart: pirmais fails
    if len(request.files):
        fs = next(iter(request.files.values()))
        return _decode_image_bgr_from_filestorage(fs)
    # 3) dataURL formā
    data_url = request.form.get("image")
    if data_url:
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", data_url)
        if m:
            raw = base64.b64decode(m.group(1))
            return _decode_image_bgr_from_bytes(raw)
    # 4) raw binary ar image/* content-type
    if request.data and request.mimetype and request.mimetype.startswith("image/"):
        return _decode_image_bgr_from_bytes(request.data)

    # ja nekas neder
    raise ValueError("Nav atrasts attēls. Sūti multipart/form-data ar lauku 'file' vai 'image', "
                     "vai dataURL 'image', vai raw image/* body.")

# ---------------- endpoints ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        img_bgr = _pick_image_from_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        strength = float(request.form.get("strength", "0.75"))
    except Exception:
        strength = 0.75
    strength = float(np.clip(strength, 0.0, 1.0))

    try:
        out_bgr = whiten_dispatch(img_bgr, strength=strength)
        return send_file(_jpeg_bytes_from_bgr(out_bgr, 92), mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": f"processing failed: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")), debug=False, threaded=True)
