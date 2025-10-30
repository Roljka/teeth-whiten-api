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

# ==== PARAMETRI ====
AI_SIZE = 768          # mazāks nekā 1024 -> lētāk
DELTA_L = 14           # cik stipri balinām
DELTA_B = -14          # cik ļoti noņemam dzeltenumu
FEATHER = 3            # malas nedaudz mīkstas
MOUTH_Y_MIN = 0.45     # mute nevar būt augstāk par šo
MOUTH_Y_MAX = 0.9      # mute nevar būt zemāk par šo
BOX_MIN_W = 0.12       # lai nav mega šaurs
BOX_MAX_W = 0.55       # lai nepaņem visu seju
BOX_MAX_H = 0.20

VISION_PROMPT = (
    "You are a vision model that ONLY detects teeth areas in human face photos. "
    "Return STRICT JSON ONLY in this exact shape:\n"
    "{\n"
    "  \"upper\": {\"x\": <float 0..1>, \"y\": <float 0..1>, \"w\": <float 0..1>, \"h\": <float 0..1>} | null,\n"
    "  \"lower\": {\"x\": <float 0..1>, \"y\": <float 0..1>, \"w\": <float 0..1>, \"h\": <float 0..1>} | null\n"
    "}\n"
    "Rules:\n"
    "- x,y is top-left corner.\n"
    "- Values MUST be normalized to [0,1].\n"
    "- If teeth are closed and not visible, return both as null.\n"
    "- DO NOT return polygons.\n"
    "- DO NOT return extra fields.\n"
)

# ==== HELPERI ====

def exif_to_rgb(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")

def to_square(img: Image.Image, size=AI_SIZE):
    """Ieliekam bilžu kvadrātā (centrēti), lai AI ir fiksēts izmērs"""
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

def clamp_box(b):
    """Noliekam AI kasti saprātīgā mutes zonā"""
    if b is None:
        return None
    x = float(b.get("x", 0.0))
    y = float(b.get("y", 0.0))
    w = float(b.get("w", 0.0))
    h = float(b.get("h", 0.0))

    # mutei jābūt noteiktā augstumā
    if y < MOUTH_Y_MIN:
        y = MOUTH_Y_MIN
    if y > MOUTH_Y_MAX:
        y = MOUTH_Y_MAX - 0.05  # mazliet uz augšu

    # izmērus normalizējam
    if w < BOX_MIN_W:
        w = BOX_MIN_W
    if w > BOX_MAX_W:
        w = BOX_MAX_W
    if h <= 0 or h > BOX_MAX_H:
        h = 0.12  # noklusētais mutes augstums

    # x robežas
    if x < 0.15:
        x = 0.15
    if x + w > 0.85:
        x = 0.85 - w

    return {"x": x, "y": y, "w": w, "h": h}

def box_to_mask(box, size=AI_SIZE, feather=FEATHER):
    """Uzzīmējam no vienas kastes ovālu masku"""
    mask = Image.new("L", (size, size), 0)
    if not box:
        return mask
    draw = ImageDraw.Draw(mask)
    x = int(box["x"] * size)
    y = int(box["y"] * size)
    w = int(box["w"] * size)
    h = int(box["h"] * size)
    # mazliet paplašinām, lai noķer visus zobus
    pad_w = int(w * 0.08)
    pad_h = int(h * 0.25)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(size, x + w + pad_w)
    y2 = min(size, y + h + pad_h)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask

def whiten_lab(img: Image.Image, mask: Image.Image,
               delta_l=DELTA_L, delta_b=DELTA_B) -> Image.Image:
    lab = img.convert("LAB")
    L, A, B = lab.split()

    L_np = np.array(L, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)

    M = np.array(mask.resize(img.size, Image.LANCZOS), dtype=np.float32) / 255.0

    dL = delta_l * 2.55
    dB = delta_b * 2.55

    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")

    out_lab = Image.merge("LAB", (L2, A, B2))
    return out_lab.convert("RGB")

def fallback_mask(square_img: Image.Image) -> Image.Image:
    """ja AI atnāca ar sviestu – mūsu deterministiskā versija"""
    w, h = square_img.size
    arr = np.array(square_img).astype(np.uint8)
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    diff = mx - mn
    sat = np.zeros_like(mx)
    nz = mx != 0
    sat[nz] = (diff[nz] / mx[nz]) * 255.0

    top = int(h * 0.6)
    bottom = int(h * 0.8)
    left = int(w * 0.25)
    right = int(w * 0.75)

    bright = mx > 165
    low_sat = sat < 80

    roi = np.zeros((h, w), dtype=bool)
    roi[top:bottom, left:right] = True

    mask_bool = bright & low_sat & roi
    mask = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    return mask

# ==== MARŠRUTI ====

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="upload with 'file'"), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
        img = exif_to_rgb(raw)
    except Exception as e:
        return jsonify(error=f"cannot read image: {e}"), 400

    # 1) ieliekam kvadrātā
    sq, orig_size, resized_size, offsets = to_square(img, AI_SIZE)

    # 2) sagatavojam AI input
    sq_b64 = base64.b64encode(png_bytes(sq)).decode("utf-8")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sq_b64}"}}
            ]
        }
    ]

    upper_box = None
    lower_box = None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=180,
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        upper_box = clamp_box(data.get("upper"))
        lower_box = clamp_box(data.get("lower"))
    except Exception:
        upper_box = None
        lower_box = None

    # 3) būvējam galīgo masku
    if upper_box or lower_box:
        mask = Image.new("L", (AI_SIZE, AI_SIZE), 0)
        if upper_box:
            m_up = box_to_mask(upper_box, AI_SIZE, FEATHER)
            mask = Image.composite(m_up, mask, m_up)
        if lower_box:
            m_low = box_to_mask(lower_box, AI_SIZE, FEATHER)
            mask = Image.composite(m_low, mask, m_low)
    else:
        mask = fallback_mask(sq)

    # 4) balinām lokāli
    whitened_sq = whiten_lab(sq, mask, DELTA_L, DELTA_B)

    # 5) atgriezamies pie oriģinālā izmēra
    final_img = back_from_square(whitened_sq, orig_size, resized_size, offsets)

    return send_file(
        io.BytesIO(png_bytes(final_img)),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
