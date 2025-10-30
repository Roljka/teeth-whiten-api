import os
import io
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# cik stipri balinām
DELTA_L = 14          # +L
DELTA_B = -14         # -dzeltenais
FEATHER = 3           # mazs blur, lai nebalina pusi sejas
SIZE = 1024           # AI redzamais izmērs

VISION_PROMPT = (
    "You are a vision model. Detect the person's VISIBLE TEETH in the image. "
    "Return STRICT JSON only. "
    "Format: {\"polygons\":[{\"points\":[[x1,y1],[x2,y2],...]}]} "
    "Coordinates MUST be normalized floats in range [0,1], relative to image width and height. "
    "Use 8-20 points per polygon. "
    "If teeth are slightly open, follow the outer tooth boundary. "
    "If you are unsure, return an empty list: {\"polygons\":[]}."
)

def exif_to_rgb(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")

def to_square(img: Image.Image, size=SIZE):
    """ieliekam 1024x1024, piefiksējam kā atgriezties"""
    w0, h0 = img.size
    img2 = img.copy()
    img2.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - img2.width) // 2
    oy = (size - img2.height) // 2
    bg.paste(img2, (ox, oy))
    return bg, (w0, h0), (img2.width, img2.height), (ox, oy)

def back_from_square(square_img: Image.Image, orig_size, resized_size, offsets):
    w0, h0 = orig_size
    rw, rh = resized_size
    ox, oy = offsets
    crop = square_img.crop((ox, oy, ox + rw, oy + rh))
    return crop.resize((w0, h0), Image.LANCZOS)

def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def polygons_to_mask(polygons, size=SIZE, feather=FEATHER):
    """
    polygons: [{'points': [[x,y],...]}] ar x,y 0..1
    atgriež PIL L masku 0..255
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    drawn = False
    for poly in polygons:
        pts = poly.get("points", [])
        if len(pts) < 3:
            continue
        pix = [(int(x * size), int(y * size)) for x, y in pts]
        draw.polygon(pix, fill=255)
        drawn = True
    if not drawn:
        return Image.new("L", (size, size), 0)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask

def whiten_lab(img: Image.Image, mask: Image.Image,
               delta_l=DELTA_L, delta_b=DELTA_B) -> Image.Image:
    """
    lokāla balināšana LAB telpā
    """
    lab = img.convert("LAB")
    L, A, B = lab.split()

    L_np = np.array(L, dtype=np.float32)
    A_np = np.array(A, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)

    M = np.array(mask.resize(img.size, Image.LANCZOS), dtype=np.float32) / 255.0

    dL = delta_l * 2.55
    dB = delta_b * 2.55

    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")

    lab2 = Image.merge("LAB", (L2, A, B2))
    out = lab2.convert("RGB")
    return out

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="upload with 'file'"), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
        img = exif_to_rgb(raw)
    except Exception as e:
        return jsonify(error=f"cannot read image: {e}"), 400

    # 1) normalizējam uz 1024
    sq, orig_size, resized_size, offsets = to_square(img, SIZE)

    # 2) sagatavojam bildi visionam
    sq_bytes = png_bytes(sq)
    sq_b64 = base64.b64encode(sq_bytes).decode("utf-8")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sq_b64}"}}
            ]
        }
    ]

    # 3) jautājam tikai koordinātes
    try:
      vis = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=messages,
          temperature=0,
          max_tokens=300,
      )
      txt = vis.choices[0].message.content.strip()
      data = json.loads(txt)
      polygons = data.get("polygons", [])
    except Exception as e:
      # ja kaut kas neizdodas – tukša maska → balināšana nenotiek
      polygons = []

    # 4) būvējam masku no koordinātēm
    mask = polygons_to_mask(polygons, size=SIZE, feather=FEATHER)

    # 5) lokāli balinām
    whitened_sq = whiten_lab(sq, mask)

    # 6) atgriežam proporciju
    final_img = back_from_square(whitened_sq, orig_size, resized_size, offsets)

    out_bytes = png_bytes(final_img)
    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
