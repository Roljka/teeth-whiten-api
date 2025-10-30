import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

app = Flask(__name__)
CORS(app)


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
    # 1) faila lauks
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # 2) ielasām bildi
    try:
        img = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # pārkodējam uz PNG
    png_bytes = pil_to_png_bytes(img)

    # izveidojam "failu" ar nosaukumu, lai openai nesaka application/octet-stream
    img_file = io.BytesIO(png_bytes)
    img_file.name = "image.png"  # <-- ŠĪ IR GALVENĀ RINDA

    try:
        client = OpenAI(api_key=api_key)

        result = client.images.edit(
            model="gpt-image-1",
            image=img_file,  # tagad tas ir "image.png"
            prompt=(
                "Whiten ONLY the visible teeth. Keep lips, gums, skin, hair and background unchanged. "
                "Natural, realistic result; no halo, no glow, no overexposure."
            ),
            size="1024x1024",
        )

        b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(b64)

    except Exception as e:
        msg = str(e)
        if "Incorrect API key provided" in msg or "invalid_api_key" in msg:
            return jsonify(error="OpenAI authentication failed: " + msg), 401
        if "unsupported_file_mimetype" in msg:
            return jsonify(error="OpenAI rejected image: unsupported mimetype. We forced PNG but server still refused."), 400
        return jsonify(error="OpenAI call failed: " + msg), 502

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
