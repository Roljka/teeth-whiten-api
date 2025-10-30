import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from openai import OpenAI

app = Flask(__name__)
CORS(app)

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
MAX_SIDE = 1024

PROMPT = (
    "Whiten ONLY the person's existing natural teeth enamel. "
    "Do NOT replace or redraw teeth. "
    "Do NOT change tooth shape, count, alignment, gums, lips, skin, beard, hair or background. "
    "Keep overall brightness and contrast unchanged. "
    "Make a subtle, realistic whitening (1-2 shades). "
    "If you cannot clearly detect the teeth, make NO changes."
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

    square_img, orig_size, resized_size, offsets = to_square_with_meta(raw, MAX_SIDE)

    # ⇩⇩⇩ ŠEIT BIJA PROBLĒMA ⇩⇩⇩
    square_png_bytes = pil_to_png_bytes(square_img)
    image_file = io.BytesIO(square_png_bytes)
    image_file.name = "image.png"   # <- svarīgi! lai nav application/octet-stream

    client = OpenAI(api_key=api_key)

    try:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=image_file,
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

    final_img = from_square_back(edited_img, orig_size, resized_size, offsets)

    out_bytes = pil_to_png_bytes(final_img)
    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )

@app.errorhandler(500)
def handle_500(e):
    return jsonify(error="Internal server error", detail=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
