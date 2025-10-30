import os
import io
import base64
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# lētais attēlu modelis
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
# lētais vision (lai tikai pateiktu kur ir mute)
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")

MAX_SIDE = 1024  # uz šo samazinām pirms sūtam
# cik liels būs ovāls ap muti (kā daļa no kvadrāta)
MOUTH_RX = 0.21   # horizontālais rādiuss
MOUTH_RY = 0.14   # vertikālais rādiuss


PROMPT_IMAGE = (
    "Whiten ONLY the person's existing natural teeth enamel. "
    "Do NOT replace or redraw the person. "
    "Do NOT change lips, skin, beard, eyes, hair or background. "
    "Keep everything outside the transparent masked area EXACTLY the same. "
    "Inside the masked area, remove yellow tint and make teeth a bit whiter, natural, realistic. "
    "If there are no clear teeth in the masked area, make NO change."
)

PROMPT_MOUTH_CENTER = (
    "You will be shown a human face photo. "
    "Return STRICT JSON ONLY with the CENTER of the visible mouth/teeth region. "
    "Format: {\"cx\": <float 0..1>, \"cy\": <float 0..1>} "
    "cx and cy must be normalized 0..1 relative to image width and height. "
    "If person is smiling upwards (camera below), still give the mouth center. "
    "If you cannot detect a mouth, return {\"cx\": 0.5, \"cy\": 0.7}."
)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_square_with_meta(img: Image.Image, size: int = MAX_SIDE):
    img = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = img.size

    img_copy = img.copy()
    img_copy.thumbnail((size, size), Image.LANCZOS)

    rw, rh = img_copy.size
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - rw) // 2
    oy = (size - rh) // 2
    bg.paste(img_copy, (ox, oy))

    return bg, (w0, h0), (rw, rh), (ox, oy)


def from_square_back(square_img: Image.Image, orig_size, resized_size, offsets):
    w0, h0 = orig_size
    rw, rh = resized_size
    ox, oy = offsets
    cropped = square_img.crop((ox, oy, ox + rw, oy + rh))
    final_img = cropped.resize((w0, h0), Image.LANCZOS)
    return final_img


def make_transparent_mask_at(cx: float, cy: float, size: int = MAX_SIDE,
                             rx: float = MOUTH_RX, ry: float = MOUTH_RY) -> Image.Image:
    """
    OpenAI image edit GRIB caurspīdīgo zonu tur, kur jāeditē.
    Tāpēc taisām RGBA:
      - viss melns un necaurspīdīgs (0,0,0,255)
      - ovāls caurspīdīgs  (0,0,0,0)
    cx, cy, rx, ry ir normalizēti (0..1)
    """
    mask = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)

    cx_px = int(cx * size)
    cy_px = int(cy * size)
    rx_px = int(rx * size)
    ry_px = int(ry * size)

    x1 = max(0, cx_px - rx_px)
    y1 = max(0, cy_px - ry_px)
    x2 = min(size, cx_px + rx_px)
    y2 = min(size, cy_px + ry_px)

    draw.ellipse([x1, y1, x2, y2], fill=(0, 0, 0, 0))  # caurspīdīgais caurums
    return mask


@app.get("/health")
def health():
    return jsonify(ok=True)


@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # 1) ieliekam kvadrātā
    square_img, orig_size, resized_size, offsets = to_square_with_meta(raw, MAX_SIDE)
    square_png = pil_to_png_bytes(square_img)

    # 2) vision: noskaidrojam mutes centru
    client = OpenAI(api_key=api_key)

    b64 = base64.b64encode(square_png).decode("utf-8")
    messages = [
        {"role": "system", "content": PROMPT_MOUTH_CENTER},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }
    ]

    cx, cy = 0.5, 0.7  # noklusētais, ja AI apmaldās
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=50,
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        cx = float(data.get("cx", 0.5))
        cy = float(data.get("cy", 0.7))
        # nedaudz noliekam saprātīgās robežās
        cx = min(max(cx, 0.20), 0.80)
        cy = min(max(cy, 0.50), 0.90)
    except Exception:
        # ja vision izgāžas – paliek noklusētais
        pass

    # 3) uztaisa masku TIEŠI tajā vietā, kur AI teica
    mask_img = make_transparent_mask_at(cx, cy, size=MAX_SIDE)

    # 4) sagatavojam failus ar nosaukumiem (lai nav octet-stream)
    image_file = io.BytesIO(square_png)
    image_file.name = "image.png"

    mask_bytes = pil_to_png_bytes(mask_img)
    mask_file = io.BytesIO(mask_bytes)
    mask_file.name = "mask.png"

    # 5) sūtam uz image edit
    try:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=image_file,
            mask=mask_file,
            prompt=PROMPT,
            size="1024x1024"
        )
    except Exception as e:
        return jsonify(error=f"OpenAI call failed: {str(e)}"), 502

    try:
        out_b64 = result.data[0].b64_json
    except Exception:
        return jsonify(error="OpenAI returned no image data"), 502

    edited_bytes = base64.b64decode(out_b64)
    edited_img = Image.open(io.BytesIO(edited_bytes)).convert("RGB")

    # 6) nogriežam atpakaļ uz oriģinālo proporciju
    final_img = from_square_back(edited_img, orig_size, resized_size, offsets)

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
