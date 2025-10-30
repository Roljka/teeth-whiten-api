import os, io, base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    # Ātrs healthcheck (nekas smags te nenotiek)
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
    # Nekādas lejupielādes/importi šeit – tikai OpenAI zvans
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    img = Image.open(request.files["file"].stream).convert("RGB")
    png_bytes = pil_to_png_bytes(img)

    client = OpenAI(api_key=api_key)
    # EDITS: balinām tikai zobus, pārējo bildi NEMAINĀM
    result = client.images.edits(
        model="gpt-image-1",
        image=png_bytes,
        prompt=(
            "Whiten ONLY the teeth. Do not modify lips, gums, skin, hair or background. "
            "Keep overall brightness/contrast unchanged. Natural, realistic result; no halo/glow."
        ),
        size="1024x1024"
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
