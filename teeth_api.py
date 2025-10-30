import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# 1) promti
MASK_PROMPT = (
    "Create a pure black-and-white mask of the photo. "
    "Paint the existing visible TEETH in pure white (#FFFFFF). "
    "Paint EVERYTHING else pure black (#000000): lips, gums, tongue, skin, beard, hair, background. "
    "Do NOT add or invent new teeth. "
    "Match the exact current teeth shape, tilt and spacing. Output only the mask."
)

WHITEN_PROMPT = (
    "Lighten ONLY the existing visible teeth in the photo, as specified by the mask. "
    "Do NOT add, replace or redraw teeth. "
    "Keep the exact tooth shape, size, spacing and gum line. "
    "Keep lips, skin, beard and background unchanged. "
    "Just brighten the enamel 1-2 shades for a natural result, keep texture and translucency. "
    "If unsure, make no change."
)


def make_square_1024(img: Image.Image, size: int = 1024) -> Image.Image:
    """
    - izlabo EXIF
    - konvertē uz RGB
    - samazina tā, lai garākā mala ir <= size
    - ieliek melnā kvadrātā size x size, lai nebūtu deformācija
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    # samazinām, saglabājot proporcijas
    img.thumbnail((size, size), Image.LANCZOS)
    # izveidojam kvadrātu
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    bg.paste(img, offset)
    return bg


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    return jsonify(ok=True)


@app.get("/")
def root():
    return jsonify(message="Teeth Whitening API (auto mask, 2-step). POST /whiten with file=<photo>.")


@app.post("/whiten")
def whiten():
    # 1) bilde obligāta
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # 2) nolasām un normalizējam UZREIZ uz 1024x1024
    try:
        raw_img = Image.open(request.files["file"].stream)
        img_1024 = make_square_1024(raw_img, 1024)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # 3) pārkodējam uz PNG un izveidojam DIVUS IO no viena un tā paša byta
    png_bytes = pil_to_png_bytes(img_1024)

    image_for_mask = io.BytesIO(png_bytes)
    image_for_mask.name = "image.png"

    image_for_whiten = io.BytesIO(png_bytes)
    image_for_whiten.name = "image.png"

    try:
        client = OpenAI(api_key=api_key)

        # ───────── 1. solis – ģenerējam masku ─────────
        mask_result = client.images.edit(
            model="gpt-image-1",
            image=image_for_mask,
            prompt=MASK_PROMPT,
            size="1024x1024",
        )
        mask_b64 = mask_result.data[0].b64_json
        mask_bytes = base64.b64decode(mask_b64)

        # ielādējam masku un “iztīrām” pelēkās vietas
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        # pārvēršam par striktu b/w
        mask_bw = mask_img.point(lambda p: 255 if p > 128 else 0)

        mask_io = io.BytesIO()
        mask_bw.save(mask_io, format="PNG")
        mask_io.seek(0)
        mask_io.name = "mask.png"

        # ───────── 2. solis – balinām ar masku ─────────
        whiten_result = client.images.edit(
            model="gpt-image-1",
            image=image_for_whiten,
            mask=mask_io,
            prompt=WHITEN_PROMPT,
            size="1024x1024",
        )

        out_b64 = whiten_result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

    except Exception as e:
        msg = str(e)
        if "Incorrect API key provided" in msg or "invalid_api_key" in msg:
            return jsonify(error="OpenAI authentication failed: " + msg), 401
        if "must be verified to use the model `gpt-image-1`" in msg:
            return jsonify(
                error="Your organization must be verified to use gpt-image-1.",
                detail=msg,
            ), 403
        if "mask size does not match image size" in msg:
            return jsonify(
                error="Auto-mask failed: mask and image must BOTH be 1024x1024. We already normalize image, so check model output.",
                detail=msg,
            ), 400
        return jsonify(error="OpenAI call failed: " + msg), 502

    # 4) sūtam atpakaļ 1024x1024 PNG
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
