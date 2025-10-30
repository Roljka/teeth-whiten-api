import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from openai import OpenAI

app = Flask(__name__)
CORS(app)

SOFT_WHITEN_PROMPT = (
    "Lighten ONLY the existing visible teeth in the photo. "
    "Do NOT add, replace or redraw teeth. "
    "Keep the exact tooth shape, size, spacing and gum line. "
    "Keep lips, skin, beard and background unchanged. "
    "Just brighten the enamel 1-2 shades for a natural result, keep texture and translucency. "
    "If unsure, make no change."
)

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    return jsonify(ok=True)


@app.get("/")
def root():
    return jsonify(message="Teeth Whitening API up. POST /whiten (multipart form: file=photo, optional mask=png).")


@app.post("/whiten")
def whiten():
    # 1) bilde obligāta
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # 2) nolasām un iztaisnojam bildi
    try:
        raw_img = Image.open(request.files["file"].stream)
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # pārkodējam uz PNG
    png_bytes = pil_to_png_bytes(img)
    img_file = io.BytesIO(png_bytes)
    img_file.name = "image.png"  # lai openai zina mimetype

    # 3) vai ir maska?
    mask_file = None
    if "mask" in request.files and request.files["mask"].filename:
        try:
            raw_mask = Image.open(request.files["mask"].stream)
            # maskai visdrošāk – 1 kanāls vai RGBA
            # ja nāk JPG, pārtaisām uz L (b/w)
            if raw_mask.mode not in ("L", "LA", "RGBA"):
                raw_mask = raw_mask.convert("L")
            mask_bytes = pil_to_png_bytes(raw_mask)
            mask_io = io.BytesIO(mask_bytes)
            mask_io.name = "mask.png"
            mask_file = mask_io
        except Exception as e:
            return jsonify(error=f"Cannot read mask: {e}"), 400

    # 4) saucam OpenAI
    try:
        client = OpenAI(api_key=api_key)

        kwargs = dict(
            model="gpt-image-1",
            image=img_file,
            prompt=SOFT_WHITEN_PROMPT,
            size="1024x1024",
        )
        if mask_file is not None:
            kwargs["mask"] = mask_file

        result = client.images.edit(**kwargs)

        b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(b64)

    except Exception as e:
        msg = str(e)
        if "Incorrect API key provided" in msg or "invalid_api_key" in msg:
            return jsonify(error="OpenAI authentication failed: " + msg), 401
        if "must be verified to use the model `gpt-image-1`" in msg:
            return jsonify(
                error="Your organization must be verified to use gpt-image-1.",
                detail=msg,
            ), 403
        if "unsupported_file_mimetype" in msg:
            return jsonify(
                error="OpenAI rejected image/mask: unsupported mimetype after PNG conversion.",
                detail=msg,
            ), 400
        return jsonify(error="OpenAI call failed: " + msg), 502

    # 5) sūtam atpakaļ bildi
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
