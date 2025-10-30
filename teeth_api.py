import os, io, base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    # vienmēr atbildēs, pat ja OPENAI nav uzlikts
    return jsonify(ok=True)


@app.get("/")
def root():
    return jsonify(message="Teeth Whitening API up. POST /whiten (multipart form, field 'file').")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # IMPORTANT: klientu veidojam ŠEIT, nevis globāli
    client = OpenAI(api_key=api_key)

    img = Image.open(request.files["file"].stream).convert("RGB")
    png_bytes = pil_to_png_bytes(img)

    # .edit (nevis .edits)
    result = client.images.edit(
        model="gpt-image-1",
        image=io.BytesIO(png_bytes),
        prompt=(
            "Whiten ONLY the visible teeth. Keep lips, gums, skin, hair and background unchanged. "
            "Natural, realistic result; no halo, no glow, no overexposure."
        ),
        size="1024x1024",
    )

    b64 = result.data[0].b64_json
    out_bytes = base64.b64decode(b64)

    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
