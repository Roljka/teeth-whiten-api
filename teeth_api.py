from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tempfile, os

# =============== Flask & CORS ===============
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =============== Globālie ielādētie resursi (modelis) ===============
# Ielādē SD inpainting modeli vienreiz (lai katrs pieprasījums nav smags)
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
pipe = pipe.to(DEVICE)

# Iesakāms CPU režīmā ierobežot pavedienus (mazāk RAM pīķu)
if DEVICE == "cpu":
    torch.set_num_threads(2)

# =============== Palīgfunkcijas ===============
def save_and_send(bgr_img, name="whitened.jpg"):
    out_path = os.path.join(tempfile.gettempdir(), name)
    cv2.imwrite(out_path, bgr_img)
    return send_file(out_path, mimetype="image/jpeg")

def resize_to_sd_multiple(img_bgr, max_side=768):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)

    # saskaņo ar 64 (SD prasība)
    def snap64(x): 
        return max(64, (x // 64) * 64)
    new_w = snap64(new_w)
    new_h = snap64(new_h)
    if new_w < 64: new_w = 64
    if new_h < 64: new_h = 64

    if (new_w, new_h) != (w, h):
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), (w, h)
    return img_bgr.copy(), None

def make_teeth_mask(bgr_img):
    """
    1) Atrodam mutes iekšpuses poligonu ar MediaPipe FaceMesh
    2) Poligonā atlasām “balto” spektru HSV (zobi) – lai izslēgtu lūpas/iemuti
    3) Tīram un mīkstinām masku
    """
    # Mediapipe ielāde lokāli funkcijā, lai neveidotu globālu TF init uzreiz
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    mask_poly = np.zeros((h, w), dtype=np.uint8)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return np.zeros((h, w), dtype=np.uint8)

        lms = res.multi_face_landmarks[0].landmark
        # Iekšējās lūpas/ mute (iekšējās kontūras): 78..87 un 308..317
        inner_idx = list(range(78, 88)) + list(range(308, 318))
        pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in inner_idx], dtype=np.int32)

        if len(pts) >= 3:
            cv2.fillPoly(mask_poly, [pts], 255)

    # HSV filtrs zobiem (gaiši, ar zemu piesātinājumu)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,   0, 160], dtype=np.uint8)
    upper = np.array([180, 60, 255], dtype=np.uint8)
    mask_color = cv2.inRange(hsv, lower, upper)

    # Kombinē: tikai mutes iekšpusē un baltajā spektrā
    mask = cv2.bitwise_and(mask_poly, mask_color)

    # Morfoloģija + blur, lai zobi būtu vienmērīgi
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)

    return mask

def prompt_for_level(level: str):
    level = (level or "natural").lower()
    if level == "hollywood":
        return "whiten teeth significantly but naturally, keep gums, lips, skin and lighting unchanged, photorealistic, high quality"
    if level == "bright":
        return "whiten teeth slightly more than natural, keep gums, lips, skin and lighting unchanged, photorealistic"
    return "whiten teeth naturally, keep gums, lips, skin and lighting unchanged, photorealistic"

# =============== Endpoints ===============
@app.route("/")
def home():
    return jsonify({"message": "Teeth Whitening API — Realistic Inpainting + Auto teeth detection 😁", "device": DEVICE})

@app.route("/health")
def health():
    try:
        _ = pipe  # ja nav izveidojies, mestu izņēmumu
        return jsonify({"ok": True, "model": MODEL_ID, "device": DEVICE})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    ai_level = request.form.get("ai_level", "natural")
    guidance = float(request.form.get("guidance", 7.0))
    steps = int(request.form.get("steps", 28))

    # ielasa attēlu
    f = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        f.save(tmp.name)
        img = cv2.imread(tmp.name)
    if img is None:
        return jsonify({"error": "Nevar nolasīt attēlu"}), 400

    # sagatavo izmēru SD modelim
    img_sd, orig_shape = resize_to_sd_multiple(img, max_side=768)

    # ģenerē automātisku zobu masku
    mask_gray = make_teeth_mask(img_sd)

    # drošībai — ja maska pārāk tukša, nelaižam cauri
    if int(cv2.countNonZero(mask_gray)) < 50:
        # ja mediapipe netrāpa, atgriežam oriģinālu (vai varam maigi pastiprināt HSV)
        return save_and_send(img, "whitened_nochange.jpg")

    # SD inpainting vajag RGB PIL + RGB masku (balts=rediģēt, melns=atstāt)
    mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(cv2.cvtColor(img_sd, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_rgb).convert("RGB")

    prompt = prompt_for_level(ai_level)

    # Inference
    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=guidance,
        num_inference_steps=steps
    ).images[0]

    out_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # atskalo uz oriģinālo izmēru, ja samazinājām
    if orig_shape is not None:
        ow, oh = orig_shape
        out_bgr = cv2.resize(out_bgr, (ow, oh), interpolation=cv2.INTER_CUBIC)

    return save_and_send(out_bgr, "whitened.jpg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
