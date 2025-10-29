import io, os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Nesāc klientu, kamēr nav vajadzīgs – bet tas ir viegls anyway
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/health")
def health():
    return jsonify(ok=True)

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

@app.post("/whiten")
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="Field 'file' missing (multipart/form-data)"), 400

        img = Image.open(request.files["file"].stream).convert("RGB")

        # Šobrīd bez lokālās maskas – izmantojam norādi promptā, lai balina tikai zobus.
        # (Ja vēlāk gribam 100% precizitāti, pievienosim masku no backend)
        png_bytes = pil_to_png_bytes(img)

        # OpenAI Images “edit” (v1)
        # Piezīme: lieto jaunāko klientu, kuram ir images.edits
        result = client.images.edits(
            model="gpt-image-1",
            image=png_bytes,
            prompt=(
                "Whiten only the TEETH. Do NOT modify lips, skin, gums or other areas. "
                "Keep brightness/contrast of the rest of the image unchanged. "
                "Natural look (no overexposure)."
            ),
            size="1024x1024"
        )

        b64 = result.data[0].b64_json
        out_bytes = io.BytesIO()
        out_bytes.write(bytes.fromhex(''))  # no-op just to keep consistent
        out_bytes = io.BytesIO(bytes.fromhex(''))  # reset
        # Konvertē no b64 → bytes
        import base64
        img_bytes = base64.b64decode(b64)

        return send_file(
            io.BytesIO(img_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="whitened.png"
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
