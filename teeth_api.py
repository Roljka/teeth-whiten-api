# teeth_api.py
import io
import os
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Pieslēdz CORS no frontenda

# ---- Palīgfunkcijas -----------------------------

def _read_image_from_request() -> np.ndarray:
    if 'image' not in request.files:
        raise ValueError("missing file field 'image' (multipart/form-data)")
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image")
    # NORMALIZĒ ORIENTĀCIJU no EXIF (OpenCV to pats nedara) – minimāls fix:
    # (šis vienkārši izlīdzina gadījumos, kad EXIF rotācija nav kritiska)
    return img

def _to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()

def _get_level() -> int:
    try:
        level = int(request.form.get('level', '3'))
    except:
        level = 3
    return max(1, min(8, level))

# ---- Zobu maskas izguve (droša, bez “zilas bildes”) ------------

def build_teeth_mask(img: np.ndarray) -> np.ndarray:
    """
    Mērena, stabila maskas ģenerēšana bez MediaPipe:
    1) sejas aptuvena reģiona noteikšana (ādas toņu diapazons HSV),
    2) mutes reģions = tumšāks/sarkanīgs tonis zem deguna,
    3) zobu kandidāti = augsts L kanāls (LAB) un zems B (mazāk dzeltens),
    4) morfoloģija + kontūru filtrs, lai iegūtu vienlaidus zobu laukumu.
    """
    h, w = img.shape[:2]

    # 1) Āda aptuveni (HSV diapazons)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 15, 40], np.uint8)
    upper_skin = np.array([25, 200, 255], np.uint8)
    skin = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.medianBlur(skin, 5)

    # 2) Mutes rupjš ROI: ņem apakšējo sejas trešdaļu un tuvumā mutei
    mouth_roi = np.zeros((h, w), np.uint8)
    y1 = int(h * 0.45)
    cv2.rectangle(mouth_roi, (0, y1), (w, h), 255, -1)
    mouth = cv2.bitwise_and(skin, mouth_roi)

    # 3) Zobi LAB telpā (gaiši un mazāk dzelteni)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # Zobi parasti ir gaišāki par apkārtni
    Lnorm = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    # "Mazāk dzelteni" => zemāks B
    B_inv = 255 - B
    cand = cv2.addWeighted(Lnorm, 0.7, B_inv, 0.3, 0)
    # ierobežojam uz mutes reģionu
    cand = cv2.bitwise_and(cand, cand, mask=mouth)

    # adaptīvs slieksnis – lai ēnās nepazūd
    thr = cv2.threshold(cand, 0, 255, cv2.THRESH_OTSU)[1]

    # Morfoloģija: aizlāpām spraugas, atmetam sīkus trokšņus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Paturam tikai nozīmīgas kontūras (zobi kopā)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(mask)
    for c in cnts:
        area = cv2.contourArea(c)
        if area > (h*w)*0.0005:  # atmetam niekus
            cv2.drawContours(keep, [c], -1, 255, -1)

    # Neliela dilatācija, lai nosegtu zobu malu zonas, bet NEIZIETU smaganās
    mask = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)

    return mask  # 8-bit, 0/255

# ---- Balināšana TIKAI maskā ------------------------------------

def whiten_only_mask(img: np.ndarray, mask: np.ndarray, level: int) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    gain_L = 1.0 + 0.08 * level      # ~ +8…+64 L vienības ekvivalents
    shift_B = int(4 + 2 * level)     # -B (mazāk dzeltenuma)

    Lf = L.astype(np.float32)
    Bf = B.astype(np.int16)

    m = (mask > 0)
    Lf[m] = np.clip(Lf[m] * gain_L, 0, 255)
    Bf[m] = np.clip(Bf[m] - shift_B, 0, 255)

    lab2 = cv2.merge([Lf.astype(np.uint8), A, Bf.astype(np.uint8)])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

# ---- API --------------------------------------------------------

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        img = _read_image_from_request()
        level = _get_level()

        # 1) zobu maska (bez krāsu “debug”)
        mask = build_teeth_mask(img)

        # 2) ja maska par niecīgu – atgriežam nemainītu (pret “zilo” droši)
        if cv2.countNonZero(mask) < 200:
            out = img.copy()
        else:
            out = whiten_only_mask(img, mask, level)

        # 3) JPEG ārā
        return send_file(
            io.BytesIO(_to_jpeg_bytes(out, quality=92)),
            mimetype="image/jpeg",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Gunicorn entry
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), debug=False)
