import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
import requests

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")


def pil_to_png_bytes(pil_img: Image.Image) -> io.BytesIO:
    """PIL → PNG bytes (ko var aizsūtīt uz OpenAI images/edits)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.route("/", methods=["GET"])
def index():
    return jsonify(status="ok", msg="Teeth Whitening API (OpenAI edition) is alive 🦷")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True)


@app.route("/whiten", methods=["POST"])
def whiten():
    if not OPENAI_API_KEY:
        return jsonify(error="OPENAI_API_KEY is not set on the server"), 500

    if "file" not in request.files:
        return jsonify(error="Field 'file' is missing. Send multipart/form-data with 'file'."), 400

    try:
        # 1) nolasa bildi
        up = request.files["file"]
        img = Image.open(up.stream)
        # salabo EXIF orientāciju (selfie mode u.tml.)
        img = ImageOps.exif_transpose(img)

        # pēc vajadzības var samazināt ļoti lielas bildes
        max_side = 1600
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img_bytes = pil_to_png_bytes(img)

        # 2) sagatavojam pieprasījumu uz OpenAI Images Edit
        # te ir ļoti agresīvs prompt – speciāli, lai nelien pie lūpām / sejas
        prompt = (
            "Whiten ONLY the visible teeth in this photo. "
            "Do NOT alter lips, skin, nose, eyes, hair, background or lighting. "
            "Keep facial features, contrast, colors and ethnicity exactly the same. "
            "Just make the teeth 20% whiter and remove yellow tint."
        )

        files = {
            "image": ("teeth.png", img_bytes, "image/png"),
        }
        data = {
            "model": OPENAI_IMAGE_MODEL,
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1,
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        resp = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers=headers,
            data=data,
            files=files,
            timeout=90,
        )

        if resp.status_code != 200:
            # atsūtām, ko tieši atbildēja OpenAI – lai var debugot frontā
            return jsonify(
                error="openai_error",
                status=resp.status_code,
                detail=resp.text,
            ), 500

        out = resp.json()
        if "data" not in out or not out["data"]:
            return jsonify(error="openai_no_image_returned", raw=out), 500

        b64_img = out["data"][0]["b64_json"]
        img_bin = base64.b64decode(b64_img)

        return send_file(
            io.BytesIO(img_bin),
            mimetype="image/png",
            as_attachment=False,
            download_name="whitened.png",
        )

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
