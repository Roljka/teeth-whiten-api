import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# 1) prompts
MASK_PROMPT = (
    "Create a pure black-and-white mask of the photo. "
    "Paint the existing visible TEETH in pure white (#FFFFFF). "
    "Paint EVERYTHING else (lips, gums, tongue, skin, beard, hair, background, clothes) pure black (#000000). "
    "Do NOT add or invent new teeth. Match the exact current teeth shape, tilt and spacing. "
    "Output only the mask."
)

WHITEN_PROMPT = (
    "Lighten ONLY the existing visible teeth in the photo, as specified by the mask. "
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
    return jsonify(message="Teeth Whitening API (auto mask). POST /whiten with multipart form-data: file=<photo>.")


@app.post("/whiten")
def whiten():
    # 1) bilde obligāta
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    # 2) nolasām un iztaisnojam orientāciju
    try:
        raw_img = Image.open(request.files["file"].stream)
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # 3) pārkodējam uz PNG (vienreiz) – pēc tam no šī byta taisīsim 2 streamus
    png_bytes = pil_to_png_bytes(img)

    # pirmais stream – maskas ģenerēšanai
    image_io_for_mask = io.BytesIO(png_bytes)
    image_io_for_mask.name = "image.png"

    # otrais stream – pašai balināšanai
    image_io_for_whiten = io.BytesIO(png_bytes)
    image_io_for_whiten.name = "image.png"

    try:
        client = OpenAI(api_key=api_key)

        # ─────────────────────────────────
        # 1. SOLIS – uzģenerējam masku no bildes
        # ─────────────────────────────────
        mask_result = client.images.edit(
            model="gpt-image-1",
            image=image_io_for_mask,
            prompt=MASK_PROMPT,
            size="1024x1024",
        )

        mask_b64 = mask_result.data[0].b64_json
        mask_bytes = base64.b64decode(mask_b64)

        # noliekam masku kā PIL, lai varam notīrīt pelēkos pikseļus
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        # threshold – viss virs 128 = balts, pārējais melns
        mask_bw = mask_img.point(lambda p: 255 if p > 128 else 0)

        # ieliekam atpakaļ kā PNG ar pareizu name
        mask_io = io.BytesIO()
        mask_bw.save(mask_io, format="PNG")
        mask_io.seek(0)
        mask_io.name = "mask.png"

        # ─────────────────────────────────
        # 2. SOLIS – balinām ar masku
        # ─────────────────────────────────
        whiten_result = client.images.edit(
            model="gpt-image-1",
            image=image_io_for_whiten,
            mask=mask_io,
            prompt=WHITEN_PROMPT,
            size="1024x1024",
        )

        out_b64 = whiten_result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

    except Exception as e:
        msg = str(e)
        # tipiskākās kļūdas
        if "Incorrect API key provided" in msg or "invalid_api_key" in msg:
            return jsonify(error="OpenAI authentication failed: " + msg), 401
        if "must be verified to use the model `gpt-image-1`" in msg:
            return jsonify(
                error="Your organization must be verified to use gpt-image-1.",
                detail=msg,
            ), 403
        if "unsupported_file_mimetype" in msg:
            return jsonify(
                error="OpenAI rejected the image or mask: unsupported mimetype.",
                detail=msg,
            ), 400
        # ja maskas solis sabrūk, varam mēģināt vienkāršo balināšanu bez maskas
        # (lai lietotājs tomēr saņem kaut ko)
        try:
            client = OpenAI(api_key=api_key)
            fallback_result = client.images.edit(
                model="gpt-image-1",
                image=io.BytesIO(png_bytes),
                prompt=WHITEN_PROMPT,
                size="1024x1024",
            )
            fb_b64 = fallback_result.data[0].b64_json
            fb_bytes = base64.b64decode(fb_b64)
            return send_file(
                io.BytesIO(fb_bytes),
                mimetype="image/png",
                as_attachment=False,
                download_name="whitened.png"
            )
        except Exception as e2:
            return jsonify(error="OpenAI call failed: " + msg, fallback_error=str(e2)), 502

    # 3) ja viss ok – sūtam bildi
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
