import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFilter

app = Flask(__name__)
CORS(app)

# cik stipri balinām
DELTA_L = 14      # gaišums
DELTA_B = -16     # mazāk dzeltenuma
FEATHER_PX = 5    # malas mīkstas

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def exif_to_rgb(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")

def to_square_1024_with_offsets(img: Image.Image, size=1024):
    """ieliekam 1024x1024, atgriežam arī offsetus un izmērus, lai var atgriezt oriģinālo proporciju"""
    w0, h0 = img.size
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    offset_x = (size - img.width) // 2
    offset_y = (size - img.height) // 2
    bg.paste(img, (offset_x, offset_y))
    return bg, (w0, h0), (img.width, img.height), (offset_x, offset_y)

def from_square_back(square_img: Image.Image, orig_size, resized_size, offsets):
    """no 1024x1024 atgriežamies pie oriģinālās proporcijas"""
    w0, h0 = orig_size
    rw, rh = resized_size
    ox, oy = offsets
    crop = square_img.crop((ox, oy, ox + rw, oy + rh))
    return crop.resize((w0, h0), Image.LANCZOS)

def build_teeth_mask_from_brightness(square_img: Image.Image,
                                     roi_top=0.55,
                                     roi_bottom=0.9,
                                     feather=FEATHER_PX):
    """
    Mēģinām atrast gaišākos, mazsātīgos pikseļus mutes zonā.
    Tas ir stabilāk nekā paļauties uz modeli.
    """
    w, h = square_img.size
    # PIL -> numpy (RGB)
    arr = np.array(square_img).astype(np.uint8)

    # aprēķinām HSV aptuveni
    # (ātri, ne superprecīzi – pietiek šim use-case)
    r = arr[:, :, 0].astype(np.float32) / 255.0
    g = arr[:, :, 1].astype(np.float32) / 255.0
    b = arr[:, :, 2].astype(np.float32) / 255.0

    mx = np.max(arr[:, :, :3], axis=2).astype(np.float32)  # 0..255 spilgtums
    mn = np.min(arr[:, :, :3], axis=2).astype(np.float32)
    diff = mx - mn
    # pieskaujam
    sat = np.zeros_like(mx)
    nonzero = mx != 0
    sat[nonzero] = (diff[nonzero] / mx[nonzero]) * 255.0  # 0..255 aptuvenā S

    # ROI
    y1 = int(h * roi_top)
    y2 = int(h * roi_bottom)
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # kritēriji zobiem:
    # - spilgti
    # - ne pārāk sātīgi (nevis lūpas)
    bright = mx > 170           # pietiekami gaišs
    low_sat = sat < 90          # nav pārāk krāsains
    roi = np.zeros_like(bright, dtype=bool)
    roi[y1:y2, :] = True

    teeth = bright & low_sat & roi

    roi_mask[teeth] = 255
    pil_mask = Image.fromarray(roi_mask, mode="L")
    if feather > 0:
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return pil_mask

def whiten_lab_in_mask(square_img: Image.Image,
                       mask_img: Image.Image,
                       delta_L=DELTA_L,
                       delta_B=DELTA_B):
    """
    balinām tikai tur, kur maska > 0
    """
    # PIL LAB
    lab = square_img.convert("LAB")
    L, A, B = lab.split()

    L_np = np.array(L, dtype=np.float32)
    A_np = np.array(A, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)

    mask_resized = mask_img.resize(square_img.size, Image.LANCZOS)
    M = np.array(mask_resized, dtype=np.float32) / 255.0

    dL = float(delta_L) * 2.55
    dB = float(delta_B) * 2.55

    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")
    lab2 = Image.merge("LAB", (L2, A, B2))
    out = lab2.convert("RGB")
    return out

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file'."), 400

    try:
        raw = Image.open(request.files["file"].stream)
        img = exif_to_rgb(raw)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # ieliekam kvadrātā, bet piefiksējam, kā atgriezties
    square, orig_size, resized_size, offsets = to_square_1024_with_offsets(img, 1024)

    # uzbūvējam zobu masku no spilgtuma mutes zonā
    mask = build_teeth_mask_from_brightness(square)

    # balinām tikai maskā
    whitened_square = whiten_lab_in_mask(square, mask)

    # atgriežamies pie oriģinālā izmēra → nebūs melnie stabi
    final_img = from_square_back(whitened_square, orig_size, resized_size, offsets)

    out_bytes = pil_to_png_bytes(final_img)
    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
