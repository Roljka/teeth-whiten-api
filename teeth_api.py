import os, io, base64, json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ==== Promti ====
VISION_PROMPT = (
    "You are a vision model. Locate the person's VISIBLE TEETH precisely. "
    "Return STRICT JSON ONLY, no extra words. "
    "Format: {\"polygons\":[{\"points\":[[x1,y1],[x2,y2],...]}]} "
    "Coordinates MUST be NORMALIZED floats in range [0,1] relative to the given image (width,height). "
    "Use 8-20 points per polygon. If unsure, return an empty list: {\"polygons\":[]}."
)

# cik stipri balinām (vari pielabot pēc gaumes)
DELTA_L = 12        # gaišāks (0..100 skalai; konvertēsim uz 0..255)
DELTA_B = -18       # mazāk dzeltenuma (b>0 => dzeltens; mīnuss = uz zilganu)
FEATHER_PX = 6      # maskas mīkstināšana (px 1024x1024 telpā)

# ==== Palīgfunkcijas ====
def make_square_1024(img: Image.Image, size: int = 1024) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    bg.paste(img, offset)
    return bg

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def b64_png(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def polygons_to_mask(polygons, size=1024, feather=FEATHER_PX) -> Image.Image:
    """Poligoni NORMALIZĒTĀS koordinātēs (0..1). Atgriež mīkstinātu L masku 0..255."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    any_drawn = False
    for poly in polygons:
        pts = poly.get("points", [])
        if len(pts) < 3: 
            continue
        pix = [(int(x*size), int(y*size)) for x, y in pts]
        draw.polygon(pix, fill=255)
        any_drawn = True
    if not any_drawn:
        return Image.new("L", (size, size), 0)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask

def whiten_in_lab(rgb_img: Image.Image, mask_img: Image.Image,
                  delta_L=DELTA_L, delta_b=DELTA_B) -> Image.Image:
    """
    Krāsu korekcija tikai maskas zonā:
    - pārejam uz LAB
    - L palielinām, b samazinām (mazāk dzeltens)
    - sapludinām ar mīkstu masku
    """
    # Pillow LAB kanāli aptuveni 0..255; L ~ spilgtums (0..100 -> 0..255)
    lab = rgb_img.convert("LAB")
    L, A, B = lab.split()

    L_np = np.array(L, dtype=np.float32)
    A_np = np.array(A, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)

    mask = mask_img.resize(rgb_img.size, Image.LANCZOS)
    M = np.array(mask, dtype=np.float32) / 255.0  # 0..1

    # pārrēķinam delta uz 0..255 skalu: 100 -> 255 ~ *2.55
    dL = float(delta_L) * 2.55
    dB = float(delta_b) * 2.55

    # pielietojam tikai maskas zonā (alpha-blend – proportional to mask)
    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")

    lab2 = Image.merge("LAB", (L2, A, B2))
    out = lab2.convert("RGB")
    return out

# ==== Maršruti ====
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

    # 1) nolasām un normalizējam
    try:
        raw_img = Image.open(request.files["file"].stream)
        img_1024 = make_square_1024(raw_img, 1024)
    except Exception as e:
        return jsonify(error=f"Cannot read image: {e}"), 400

    png_bytes = pil_to_png_bytes(img_1024)
    img_b64 = b64_png(png_bytes)

    # 2) VISION: iegūstam zobu poligonus (normalized 0..1)
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return polygons JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ],
        },
    ]

    try:
        vis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=250,
        )
        txt = vis.choices[0].message.content.strip()
        data = json.loads(txt)
        polygons = data.get("polygons", [])
    except Exception:
        # drošs rezerves variants – tukša maska (nekas netiks mainīts)
        polygons = []

    # 3) būvējam masku un taisām L*a*b* korekciju lokāli
    mask = polygons_to_mask(polygons, size=1024, feather=FEATHER_PX)
    out_img = whiten_in_lab(img_1024, mask, delta_L=DELTA_L, delta_b=DELTA_B)

    # 4) atbilde kā PNG
    out_bytes = pil_to_png_bytes(out_img)
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
