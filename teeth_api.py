import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# stingrais prompts – tikai balinām, neko nepārvelkam
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
    return jsonify(message="Teeth Whitening API up. POST /whiten (multipart form, field 'file').")


@app.post("/whiten")
def whiten():
    # 1) pārbaudām failu
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # 2) nolasām un iztaisnojam orientāciju
    try:
        raw_img = Image.open(request.files["file"].stream)
        # šis pārvērtīs EXIF orientāciju par reāliem pikseļiem
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # 3) pārkodējam uz PNG
    png_bytes = pil_to_png_bytes(img)

    # 4) sataisām faila objektu ar nosaukumu, lai openai nedusmojas par mimetype
    img_file = io.BytesIO(png_bytes)
    img_file.name = "image.png"

    # 5) saucam OpenAI
    try:
        client = OpenAI(api_key=api_key)

        result = client.images.edit(
            model="gpt-image-1",
            image=img_file,
            prompt=SOFT_WHITEN_PROMPT,
            size="1024x1024",
        )

        b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(b64)

    except Exception as e:
        msg = str(e)
        # tipiskās kļūdas
        if "Incorrect API key provided" in msg or "invalid_api_key" in msg:
            return jsonify(error="OpenAI authentication failed: " + msg), 401
        if "must be verified to use the model `gpt-image-1`" in msg:
            return jsonify(
                error="Your organization must be verified to use gpt-image-1.",
                detail=msg,
            ), 403
        if "unsupported_file_mimetype" in msg:
            return jsonify(
                error="OpenAI rejected image: unsupported mimetype after PNG conversion.",
                detail=msg,
            ), 400
        # pārējās
        return jsonify(error="OpenAI call failed: " + msg), 502

    # 6) viss ok – sūtām bildi
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
