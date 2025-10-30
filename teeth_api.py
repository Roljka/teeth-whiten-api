import os
import io
import base64
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# promti
VISION_PROMPT = (
    "You are a vision model. Find the person's TEETH in this photo. "
    "Return STRICT JSON ONLY, no explanation. "
    "Format: {\"boxes\": [{\"x\": <int>, \"y\": <int>, \"w\": <int>, \"h\": <int>}]}. "
    "Coordinates are in pixels in the given image resolution. "
    "x,y is top-left corner. "
    "Return 1 box if you are unsure. "
    "If you really cannot find teeth, return {\"boxes\": []}."
)

WHITEN_PROMPT = (
    "Lighten ONLY the existing visible teeth in the photo, as specified by the mask. "
    "Do NOT add, replace or redraw teeth. "
    "Keep the exact tooth shape, size, spacing and gum line. "
    "Keep lips, skin, beard and background unchanged. "
    "Just brighten the enamel 1-2 shades for a natural result, keep texture and translucency. "
    "If unsure, make no change."
)

def make_square_1024(img: Image.Image, size: int = 1024) -> Image.Image:
    """EXIF → RGB → ieliekam 1024x1024 bez izstiepšanas (letterbox)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    bg.paste(img, offset)
    return bg

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def b64_png(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def build_mask_from_boxes(boxes, size=1024) -> io.BytesIO:
    """Uztaisām melnu masku ar baltiem zobu laukumiem."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for b in boxes:
        x = int(b.get("x", 0))
        y = int(b.get("y", 0))
        w = int(b.get("w", 0))
        h = int(b.get("h", 0))
        # drošības rezerves, lai ietilpst visi zobi
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(size, x + w + pad_x)
        y2 = min(size, y + h + pad_y)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    mask_io = io.BytesIO()
    mask.save(mask_io, format="PNG")
    mask_io.seek(0)
    mask_io.name = "mask.png"
    return mask_io


@app.get("/health")
def health():
    return jsonify(ok=True)


@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="Upload with field 'file' (multipart/form-data)."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set on the server."), 500

    try:
        raw_img = Image.open(request.files["file"].stream)
        img_1024 = make_square_1024(raw_img, 1024)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    # šos pašus 1024x1024 izmantosim visur
    png_bytes = pil_to_png_bytes(img_1024)
    img_io_for_edit = io.BytesIO(png_bytes)
    img_io_for_edit.name = "image.png"

    # ===== 1) VISION zvans: dabūt zobu kasti =====
    client = OpenAI(api_key=api_key)

    # padodam bildi kā data-url, lai nav jātur publisks links
    img_b64 = b64_png(png_bytes)
    vision_messages = [
        {
            "role": "system",
            "content": VISION_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Find the teeth and return JSON."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                }
            ]
        }
    ]

    try:
        vision_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=vision_messages,
            max_tokens=200,
            temperature=0,
        )
        vision_text = vision_resp.choices[0].message.content.strip()
        # mēģinām JSON
        data = json.loads(vision_text)
        boxes = data.get("boxes", [])
    except Exception:
        # ja nesanāk – pieņemam 1 kasti sejas vidū (drošs defaults)
        boxes = [{
            "x": 1024 // 3,
            "y": 1024 // 2,
            "w": 1024 // 3,
            "h": 1024 // 5,
        }]

    # uztaisi masku no boxiem
    mask_io = build_mask_from_boxes(boxes, 1024)

    # ===== 2) IMAGE EDIT zvans =====
    try:
        edit_result = client.images.edit(
            model="gpt-image-1",
            image=img_io_for_edit,
            mask=mask_io,
            prompt=WHITEN_PROMPT,
            size="1024x1024",
        )
        out_b64 = edit_result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)
    except Exception as e:
        msg = str(e)
        if "must be verified to use the model `gpt-image-1`" in msg:
            return jsonify(error="Your org must be verified for gpt-image-1.", detail=msg), 403
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
