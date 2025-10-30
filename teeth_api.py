import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# konfigurējams
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
MAX_SIDE = 1024  # uz šo liekam kvadrātā pirms sūtam uz OpenAI

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
    """
    1) iztaisnojam pēc EXIF
    2) ieliekam 1024×1024 (vai norādīto) centrā – LETTERBOX, nevis izstiepšana
    3) atgriežam:
       - square_img: tas, ko sūtam uz OpenAI
       - orig_size: (w0, h0) – tava sākotnējā bilde
       - resized_size: (rw, rh) – cik liels fragments tika ielikts kvadrātā
       - offsets: (ox, oy) – cik no augšas/kreisās ielikts
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = img.size

    # samazinam, lai lielākā mala <= size
    img_copy = img.copy()
    img_copy.thumbnail((size, size), Image.LANCZOS)

    rw, rh = img_copy.size
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - rw) // 2
    oy = (size - rh) // 2
    bg.paste(img_copy, (ox, oy))

    return bg, (w0, h0), (rw, rh), (ox, oy)


def from_square_back(square_img: Image.Image, orig_size, resized_size, offsets):
    """
    No 1024×1024 (kur vidū stāv tava bilde) izgriežam ārā tieši to pašu laukumu
    un uzskalojam atpakaļ uz sākotnējo izmēru – lai frontā abi ir vienādi.
    """
    w0, h0 = orig_size
    rw, rh = resized_size
    ox, oy = offsets

    # izgriežam laukumu, kur bija sākotnējā bilde
    cropped = square_img.crop((ox, oy, ox + rw, oy + rh))
    # un uzliekam atpakaļ sākotnējo izmēru
    final_img = cropped.resize((w0, h0), Image.LANCZOS)
    return final_img


@app.get("/health")
def health():
    return jsonify(ok=True)


@app.post("/whiten")
def whiten():
    # 1) failam ir jābūt
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # 2) ieliekam kvadrātā un piefiksējam, kā atgriezties
    square_img, orig_size, resized_size, offsets = to_square_with_meta(raw, MAX_SIDE)
    square_png = pil_to_png_bytes(square_img)

    # 3) zvanam OpenAI ar lētāko image modeli
    client = OpenAI(api_key=api_key)

    try:
        result = client.images.edit(
            model=IMAGE_MODEL,         # <- gpt-image-1-mini
            image=square_png,
            prompt=PROMPT,
            size="1024x1024"
        )
    except Exception as e:
        # ja OpenAI nogāzās – atgriežam skaidru kļūdu, nevis HTML
        return jsonify(error=f"OpenAI call failed: {str(e)}"), 502

    try:
        b64 = result.data[0].b64_json
    except Exception:
        return jsonify(error="OpenAI returned no image data"), 502

    edited_bytes = base64.b64decode(b64)
    edited_img = Image.open(io.BytesIO(edited_bytes)).convert("RGB")

    # 4) izgriežam atpakaļ tādu pašu izmēru kā oriģinālam
    final_img = from_square_back(edited_img, orig_size, resized_size, offsets)

    # 5) sūtam bildi atpakaļ
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
