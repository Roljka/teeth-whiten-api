import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFilter

app = Flask(__name__)
CORS(app)

# balināšanas stiprums
DELTA_L = 14
DELTA_B = -14
GAUSS_PX = 2  # ļoti mazs, lai nebalinātu ādu ap muti

# ---------------- helperi ----------------

def exif_to_rgb(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")

def to_square_with_meta(img: Image.Image, size=1024):
    """ieliekam centrā 1024x1024, lai viegli rēķināt procentus"""
    w0, h0 = img.size
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - img.width) // 2
    oy = (size - img.height) // 2
    bg.paste(img, (ox, oy))
    return bg, (w0, h0), (img.width, img.height), (ox, oy)

def back_to_original(square_img: Image.Image, orig_size, resized_size, offsets):
    w0, h0 = orig_size
    rw, rh = resized_size
    ox, oy = offsets
    crop = square_img.crop((ox, oy, ox + rw, oy + rh))
    return crop.resize((w0, h0), Image.LANCZOS)

def pil_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def largest_component(mask_bool: np.ndarray):
    """
    Atrodam lielāko TRUE reģionu 2D masīvā (super vienkāršs BFS).
    Atgriežam boolean masīvu tikai ar lielāko gabalu.
    """
    h, w = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    best_mask = np.zeros_like(mask_bool, dtype=bool)
    best_size = 0

    # 4-neighbour
    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or visited[y, x]:
                continue
            # jauns komponents
            stack = [(y, x)]
            visited[y, x] = True
            current_pixels = []
            while stack:
                cy, cx = stack.pop()
                current_pixels.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(current_pixels) > best_size:
                best_size = len(current_pixels)
                best_mask[:] = False
                for py, px in current_pixels:
                    best_mask[py, px] = True

    return best_mask

def build_teeth_mask(square_img: Image.Image) -> Image.Image:
    """
    Precīzāka maska:
    - skatāmies tikai šaurā mutes joslā (piem., 58%..80%)
    - paņemam gaišus un mazsātīgus pikseļus
    - no tiem ņemam tikai LIELĀKO komponenti
    - nedaudz paplašinām
    """
    w, h = square_img.size
    arr = np.array(square_img).astype(np.uint8)

    # spilgtums
    mx = arr.max(axis=2).astype(np.float32)  # 0..255
    mn = arr.min(axis=2).astype(np.float32)
    diff = mx - mn
    sat = np.zeros_like(mx)
    nz = mx != 0
    sat[nz] = (diff[nz] / mx[nz]) * 255.0  # aptuvenā piesātinātība

    # mutes josla — šaurāka!
    top = int(h * 0.60)
    bottom = int(h * 0.78)
    # horizontāli — vidus josla
    left = int(w * 0.20)
    right = int(w * 0.80)

    region = np.zeros((h, w), dtype=bool)
    region[top:bottom, left:right] = True

    # zobi: gaiši + maza sātība + mutes reģions
    bright = mx > 165
    low_sat = sat < 80
    prelim = bright & low_sat & region

    # ja vispār neko neatradām – atgriežam tukšu
    if not prelim.any():
        return Image.new("L", (w, h), 0)

    # paņemam LIELĀKO plankumu
    comp = largest_component(prelim)

    # pārvēršam par PIL
    mask = Image.fromarray((comp * 255).astype(np.uint8), mode="L")

    # nedaudz paplašinām (dilate) ar blur
    if GAUSS_PX > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=GAUSS_PX))

    return mask

def whiten_lab(square_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    lab = square_img.convert("LAB")
    L, A, B = lab.split()

    L_np = np.array(L, dtype=np.float32)
    A_np = np.array(A, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)

    M = np.array(mask_img.resize(square_img.size, Image.LANCZOS), dtype=np.float32) / 255.0

    dL = DELTA_L * 2.55
    dB = DELTA_B * 2.55

    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")

    out_lab = Image.merge("LAB", (L2, A, B2))
    out_rgb = out_lab.convert("RGB")
    return out_rgb

# ---------------- Flask ----------------

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="upload with 'file'"), 400

    try:
        raw = Image.open(request.files["file"].stream)
        img = exif_to_rgb(raw)
    except Exception as e:
        return jsonify(error=f"cannot read image: {e}"), 400

    # 1) ieliekam kvadrātā
    square, orig_size, resized_size, offsets = to_square_with_meta(img, 1024)

    # 2) uzbūvējam precīzāku masku
    mask = build_teeth_mask(square)

    # 3) balinām tikai masku
    whitened_square = whiten_lab(square, mask)

    # 4) atgriežam uz oriģinālo proporciju
    final_img = back_to_original(whitened_square, orig_size, resized_size, offsets)

    out_bytes = pil_png_bytes(final_img)
    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
