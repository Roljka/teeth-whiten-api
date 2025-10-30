import os
import io
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFilter

from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ===== konfigurācija =====
AI_SIZE = 512           # mazāks = lētāk
DELTA_L = 14            # balināšanas stiprums (L kanāls)
DELTA_B = -14           # mazāk dzeltenuma
FEATHER = 2             # maskas mīkstināšana
MOUTH_TOP = 0.58        # pēc rotācijas – kur sākas mutes josla
MOUTH_BOTTOM = 0.82     # pēc rotācijas – kur beidzas mutes josla

ROT_PROMPT = (
    "You are a vision assistant. Your ONLY job: tell me how much to rotate the image "
    "CLOCKWISE so that the person's TEETH line (the row of visible teeth) becomes horizontal. "
    "Return STRICT JSON only, like: {\"angle_deg\": 12.5}\n"
    "- Positive = rotate clockwise\n"
    "- Negative = rotate counter-clockwise\n"
    "- If teeth already horizontal, return {\"angle_deg\": 0}"
)


# ===== palīgfunkcijas =====
def exif_to_rgb(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")


def resize_for_ai(img: Image.Image, size=AI_SIZE) -> Image.Image:
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - img.width) // 2
    oy = (size - img.height) // 2
    bg.paste(img, (ox, oy))
    return bg


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_teeth_mask_rotated(rot_img: Image.Image) -> Image.Image:
    """
    Šī ir mūsu “labākā” lokālā maska, ko tu agrāk redzēji:
    - skatāmies mutes joslu (MOUTH_TOP..MOUTH_BOTTOM)
    - ņemam gaišus + mazsātīgus pikseļus
    - ņemam tikai LIELĀKO komponenti
    - nedaudz izpludinām
    Un tagad tas strādā daudz labāk, jo bilde jau ir “taisna”.
    """
    w, h = rot_img.size
    arr = np.array(rot_img).astype(np.uint8)

    # spilgtums/sātums
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    diff = mx - mn
    sat = np.zeros_like(mx)
    nz = mx != 0
    sat[nz] = (diff[nz] / mx[nz]) * 255.0

    y1 = int(h * MOUTH_TOP)
    y2 = int(h * MOUTH_BOTTOM)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)

    roi = np.zeros((h, w), dtype=bool)
    roi[y1:y2, x1:x2] = True

    bright = mx > 165
    low_sat = sat < 85

    prelim = bright & low_sat & roi

    # ja neko neatradām – tukša maska
    if not prelim.any():
        return Image.new("L", (w, h), 0)

    # ņemam lielāko komponenti
    comp = _largest_component(prelim)
    mask = Image.fromarray((comp * 255).astype(np.uint8), mode="L")
    if FEATHER > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=FEATHER))
    return mask


def _largest_component(mask_bool: np.ndarray) -> np.ndarray:
    h, w = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    best = np.zeros_like(mask_bool, dtype=bool)
    best_size = 0

    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            curr = []
            while stack:
                cy, cx = stack.pop()
                curr.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(curr) > best_size:
                best_size = len(curr)
                best[:] = False
                for py, px in curr:
                    best[py, px] = True
    return best


def whiten_lab(img: Image.Image, mask: Image.Image,
               delta_l=DELTA_L, delta_b=DELTA_B) -> Image.Image:
    lab = img.convert("LAB")
    L, A, B = lab.split()
    L_np = np.array(L, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)
    M = np.array(mask.resize(img.size, Image.LANCZOS), dtype=np.float32) / 255.0

    dL = delta_l * 2.55
    dB = delta_b * 2.55

    L_np = np.clip(L_np + dL * M, 0, 255)
    B_np = np.clip(B_np + dB * M, 0, 255)

    L2 = Image.fromarray(L_np.astype(np.uint8), mode="L")
    B2 = Image.fromarray(B_np.astype(np.uint8), mode="L")
    out_lab = Image.merge("LAB", (L2, A, B2))
    return out_lab.convert("RGB")


def rotate_keep_canvas(img: Image.Image, angle: float) -> Image.Image:
    """
    Rotējam ar expand=True, lai nekas neapgriežas.
    """
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def center_crop_to(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """No lielākas canvasa izgriežam centru uz norādīto izmēru."""
    tw, th = target_size
    w, h = img.size
    left = (w - tw) // 2
    top = (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


# ===== FLASK =====

@app.get("/health")
def health():
    return jsonify(ok=True)


@app.post("/whiten")
def whiten():
    if "file" not in request.files:
        return jsonify(error="upload with field 'file'"), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="OPENAI_API_KEY is not set"), 500

    try:
        raw = Image.open(request.files["file"].stream)
        orig = exif_to_rgb(raw)
    except Exception as e:
        return jsonify(error=f"cannot read image: {e}"), 400

    # 1) sagatavo AI versiju (mazu kvadrātu)
    ai_img = resize_for_ai(orig, AI_SIZE)
    ai_b64 = base64.b64encode(pil_to_png_bytes(ai_img)).decode("utf-8")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": ROT_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return JSON only."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ai_b64}"}}
            ]
        }
    ]

    angle = 0.0
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=40,
        )
        txt = r.choices[0].message.content.strip()
        data = json.loads(txt)
        angle = float(data.get("angle_deg", 0.0))
    except Exception:
        angle = 0.0  # ja nekas nesanāca – balinām kā ir

    # 2) pagriežam oriģinālo bildi PRETĒJĀ virzienā (lai zobi būtu horizontāli)
    rot = rotate_keep_canvas(orig, -angle)

    # 3) uzrotētajā bildē uzbūvējam zobu masku
    mask_rot = build_teeth_mask_rotated(rot)

    # 4) balinām uzrotēto bildi
    rot_whitened = whiten_lab(rot, mask_rot, DELTA_L, DELTA_B)

    # 5) pagriežam atpakaļ
    back = rotate_keep_canvas(rot_whitened, angle)

    # 6) izgriežam atpakaļ uz oriģinālo izmēru
    final_img = center_crop_to(back, orig.size)

    return send_file(
        io.BytesIO(pil_to_png_bytes(final_img)),
        mimetype="image/png",
        as_attachment=False,
        download_name="whitened.png"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
