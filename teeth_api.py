import os, io, base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# izveidojam klientu vienreiz
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@app.get("/health")
def health():
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

    if client is None:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # ielasām bildi
    img = Image.open(request.files["file"].stream).convert("RGB")
    png_bytes = pil_to_png_bytes(img)

    # te notiek pats edits – PIEVĒRS UZMANĪBU: .edit nevis .edits !
    # oficiālajos piemēros gpt-image-1 izmanto tieši client.images.edit(...) :contentReference[oaicite:2]{index=2}
    result = client.images.edit(
        model="gpt-image-1",
        image=io.BytesIO(png_bytes),
        prompt=(
            "Whiten ONLY the visible teeth. Keep lips, gums, skin, hair and background exactly as in the original. "
            "Natural, realistic whiteness, no glow, no overexposure, no skin smoothing."
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
