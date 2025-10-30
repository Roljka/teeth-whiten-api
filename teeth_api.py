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

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")

MAX_SIDE = 1024

# ovāla izmēri (pielāgojami)
MOUTH_RX = 0.23   # platāks
MOUTH_RY = 0.15   # bišķi zemāks

PROMPT_IMAGE = (
    "Whiten ONLY the person's existing natural teeth enamel. "
    "Do NOT replace, redraw or regenerate the person, head, eyes, hair, beard or background. "
    "Do NOT change lips or mouth shape. "
    "Keep everything OUTSIDE the masked transparent region EXACTLY the same. "
    "Inside the masked area, remove yellow tint and make the enamel whiter, but natural. "
    "If there are no teeth in the mask, make NO change."
)

PROMPT_MOUTH_CENTER = (
    "You will be shown a human face photo. "
    "Return STRICT JSON ONLY with the CENTER of the visible mouth/teeth region. "
    "Format: {\"cx\": <float 0..1>, \"cy\": <float 0..1>} "
    "Values must be normalized to [0,1] relative to image width/height. "
    "If you cannot detect a mouth, return {\"cx\": 0.5, \"cy\": 0.72}."
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
    return cropped.resize((w0, h0), Image.LANCZOS)


def make_transparent_mask_at(
    cx: float,
    cy: float,
    size: int = MAX_SIDE,
    rx: float = MOUTH_RX,
    ry: float = MOUTH_RY,
) -> Image.Image:
    """
    Svarīgā izmaiņa:
    - FONS = BALTS ar alfa 255  -> (255,255,255,255)
    - Ovāls = caurspīdīgs       -> (255,255,255,0)
    Ja OpenAI nomizo alfa, viņš redzēs baltu, nevis melnu.
    """
    mask = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)

    cx_px = int(cx * size)
    cy_px = int(cy * size)
    rx_px = int(rx * size)
    ry_px = int(ry * size)

    x1 = max(0, cx_px - rx_px)
    y1 = max(0, cy_px - ry_px)
    x2 = min(size, cx_px + rx_px)
    y2 = min(size, cy_px + ry_px)

    # caurspīdīgais caurums
    draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255, 0))
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

    # 1) uztaisām kvadrātu
    square_img, orig_size, resized_size, offsets = to_square_with_meta(raw, MAX_SIDE)
    square_png = pil_to_png_bytes(square_img)

    client = OpenAI(api_key=api_key)

    # 2) vision – kur ir mute
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

    cx, cy = 0.5, 0.72  # noklusētais
    try:
        vis = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=50,
        )
        txt = vis.choices[0].message.content.strip()
        data = json.loads(txt)
        cx = float(data.get("cx", 0.5))
        cy = float(data.get("cy", 0.72))
    except Exception:
        pass

    # clamp – lai neaizšauj uz aci
    cx = min(max(cx, 0.15), 0.85)
    # dažās tavās bildēs mute ir zemāk, tāpēc nedaudz pabīdam lejup:
    cy = min(max(cy + 0.03, 0.55), 0.95)

    # 3) maska tieši tajā vietā
    mask_img = make_transparent_mask_at(cx, cy, size=MAX_SIDE)

    # 4) sagatavojam failus ar .name
    image_file = io.BytesIO(square_png)
    image_file.name = "image.png"

    mask_bytes = pil_to_png_bytes(mask_img)
    mask_file = io.BytesIO(mask_bytes)
    mask_file.name = "mask.png"

    # 5) image edit
    try:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=image_file,
            mask=mask_file,
            prompt=PROMPT_IMAGE,
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

    # 6) atpakaļ uz oriģinālo izmēru
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
