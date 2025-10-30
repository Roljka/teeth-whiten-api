import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw
from openai import OpenAI

app = Flask(__name__)
CORS(app)

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
MAX_SIDE = 1024  # uz šo liekam kvadrātā

PROMPT = (
    "Whiten ONLY the teeth that are visible in this photo. "
    "Do NOT replace, redraw or regenerate the face, head, eyes, hair, background or lighting. "
    "Do NOT change lip color or mouth shape. "
    "Keep the original person exactly the same. "
    "Inside the masked region, remove yellow tint from the existing teeth enamel and make it whiter, "
    "but natural and realistic. If there are no clear teeth inside the mask, make NO change."
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

def make_mouth_mask(size=MAX_SIDE,
                    x_min=0.28, x_max=0.72,
                    y_min=0.58, y_max=0.80):
    """
    Uztaisām L masku (balts = drīkst mainīt) tikai mutes joslai.
    Koordinātes ir normalizētas (0..1) pret kvadrātu.
    Vajadzības gadījumā šos skaitļus var pievilkt klāt/vaļā.
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    x1 = int(x_min * size)
    y1 = int(y_min * size)
    x2 = int(x_max * size)
    y2 = int(y_max * size)
    # ovāls, nevis kvadrāts -> mazāk ķer lūpas
    draw.ellipse([x1, y1, x2, y2], fill=255)
    return mask

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file'."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # 1) ieliekam kvadrātā
    square_img, orig_size, resized_size, offsets = to_square_with_meta(raw, MAX_SIDE)

    # 2) uztaisām masku tajā pašā izmērā
    mask_img = make_mouth_mask(size=MAX_SIDE)

    # 3) sagatavojam failus OpenAI (ABIEM jābūt ar name!)
    img_bytes = pil_to_png_bytes(square_img)
    img_file = io.BytesIO(img_bytes)
    img_file.name = "image.png"

    mask_bytes = pil_to_png_bytes(mask_img)
    mask_file = io.BytesIO(mask_bytes)
    mask_file.name = "mask.png"

    client = OpenAI(api_key=api_key)

    try:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=img_file,
            mask=mask_file,
            prompt=PROMPT,
            size="1024x1024"
        )
    except Exception as e:
        return jsonify(error=f"OpenAI call failed: {str(e)}"), 502

    try:
        b64 = result.data[0].b64_json
    except Exception:
        return jsonify(error="OpenAI returned no image data"), 502

    edited_bytes = base64.b64decode(b64)
    edited_img = Image.open(io.BytesIO(edited_bytes)).convert("RGB")

    # 4) nogriežam atpakaļ uz oriģinālo proporciju
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
