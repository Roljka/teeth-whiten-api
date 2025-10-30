import os
import io
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from openai import OpenAI

app = Flask(__name__)
CORS(app)

SIZE = 1024

# cik stipri balinām
DELTA_L = 14
DELTA_B = -14
FEATHER = 3   # neliels blur

# mutes "normālā zona" kvadrātā
Y_MIN = 0.45   # zem šī nav mute
Y_MAX = 0.9    # virs šī nav mute
X_MIN = 0.15   # pa labi/pa kreisi ne pārāk daudz
X_MAX = 0.85

VISION_PROMPT = (
    "You are a vision model. Detect the person's MOUTH/TEETH area in the image. "
    "Return STRICT JSON ONLY. "
    "Format: {\"box\": {\"x\": <float>, \"y\": <float>, \"w\": <float>, \"h\": <float>}} "
    "All values MUST be normalized to [0,1] relative to the image width and height. "
    "x,y = top-left. "
    "If you are unsure, return {\"box\": null}."
)

def exif_to_rgb(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")

def to_square(img: Image.Image, size=SIZE):
    """ieliekam 1024x1024, atceramies kā atgriezties"""
    w0, h0 = img.size
    img2 = img.copy()
    img2.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - img2.width) // 2
    oy = (size - img2.height) // 2
    bg.paste(img2, (ox, oy))
    return bg, (w0, h0), (img2.width, img2.height), (ox, oy)

def back_from_square(square_img: Image.Image, orig_size, resized_size, offsets):
    w0, h0 = orig_size
    rw, rh = resized_size
    ox, oy = offsets
    crop = square_img.crop((ox, oy, ox + rw, oy + rh))
    return crop.resize((w0, h0), Image.LANCZOS)

def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def build_ellipse_mask_from_box(box, size=SIZE, feather=FEATHER):
    """
    box: dict with x,y,w,h normalized
    uzzīmējam elipsi šajā taisnstūrī
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    x = int(box["x"] * size)
    y = int(box["y"] * size)
    w = int(box["w"] * size)
    h = int(box["h"] * size)
    draw.ellipse([x, y, x + w, y + h], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask

def fallback_brightness_mask(square_img: Image.Image) -> Image.Image:
    """ja AI atnes sviestu – taisām mūsu deterministisko masku ap muti"""
    w, h = square_img.size
    arr = np.array(square_img).astype(np.uint8)
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    diff = mx - mn
    sat = np.zeros_like(mx)
    nz = mx != 0
    sat[nz] = (diff[nz] / mx[nz]) * 255.0

    top = int(h * 0.60)
    bottom = int(h * 0.78)
    left = int(w * 0.20)
    right = int(w * 0.80)

    region = np.zeros((h, w), dtype=bool)
    region[top:bottom, left:right] = True

    bright = mx > 165
    low_sat = sat < 80
    prelim = bright & low_sat & region
    mask = Image.fromarray((prelim * 255).astype(np.uint8), mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    return mask

def whiten_lab(img: Image.Image, mask: Image.Image,
               delta_l=DELTA_L, delta_b=DELTA_B) -> Image.Image:
    lab = img.convert("LAB")
    L, A, B = lab.split()
    L_np = np.array(L, dtype=np.float32)
    A_np = np.array(A, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)

    M = np.array(mask.resize(img.size, Image.LANCZOS), dtype=np.float32) / 255.0

    dL = delta_l * 2.55
    dB = delta_b * 2.55

    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")

    out_lab = Image.merge("LAB", (L2, A, B2))
    out_rgb = out_lab.convert("RGB")
    return out_rgb

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="upload with 'file'"), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
        img = exif_to_rgb(raw)
    except Exception as e:
        return jsonify(error=f"cannot read image: {e}"), 400

    # 1) 1024 kvadrāts
    sq, orig_size, resized_size, offsets = to_square(img, SIZE)

    # 2) sagatavojam vision input
    sq_b64 = base64.b64encode(png_bytes(sq)).decode("utf-8")
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sq_b64}"}}
            ],
        },
    ]

    box = None
    try:
        vis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=120,
        )
        txt = vis.choices[0].message.content.strip()
        data = json.loads(txt)
        box = data.get("box")
    except Exception:
        box = None

    # 3) validējam box
    mask = None
    if box:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))

        # clamp uz mutes zonu
        # ja AI aizšāvis uz aci – iemetam atpakaļ
        if y < Y_MIN:
            y = Y_MIN
        if y + h > Y_MAX:
            h = max(0.05, Y_MAX - y)
        if x < X_MIN:
            x = X_MIN
        if x + w > X_MAX:
            w = max(0.05, X_MAX - x)

        # ja kaste ir pārāk liela (piem., acs+degunam) – metam ārā
        if h > 0.4 or w > 0.7:
            mask = None
        else:
            mask = build_ellipse_mask_from_box({"x": x, "y": y, "w": w, "h": h}, size=SIZE, feather=FEATHER)

    # ja maska nav – fallback
    if mask is None:
        mask = fallback_brightness_mask(sq)

    # 4) balinām lokāli
    whitened_sq = whiten_lab(sq, mask)

    # 5) atgriežamies pie oriģinālā izmēra
    final_img = back_from_square(whitened_sq, orig_size, resized_size, offsets)

    return send_file(
        io.BytesIO(png_bytes(final_img)),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
