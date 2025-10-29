from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tempfile, os

app = Flask(__name__)
CORS(app)

# --------------------------
# UTIL: droši saglabā un atgriež failu
# --------------------------
def _save_and_send(img, name="output.jpg"):
    out_path = os.path.join(tempfile.gettempdir(), name)
    cv2.imwrite(out_path, img)
    return send_file(out_path, mimetype="image/jpeg")

# --------------------------
# FAST režīms: HSV pikseļu balināšana tikai “zobu baltajā” spektrā
# --------------------------
def whiten_fast(img, intensity=25):
    # Konvertē uz HSV: zobi = gaiši ar zemu S
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Šie sliekšņi ir konservatīvi, lai neaiztiktu ādu/lūpas
    lower = np.array([0,   0, 170], dtype=np.uint8)
    upper = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Tikai mutes rajonam – mēģinām šaurināt ar sejas/mutes aproksimāciju (fallback, ja nav mediapipe)
    # Mazs morfoloģiskais tīrīšanas solis
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (15, 15), 6)

    out = img.copy()
    # Palielinām tikai Luma (praktiski – pieplusojam RGB vienādi)
    add = np.zeros_like(out)
    add[:, :, :] = intensity
    out[mask > 0] = cv2.add(out[mask > 0], add[mask > 0])
    return out

# --------------------------
# AI režīms: Mediapipe maska + Stable Diffusion Inpainting
# --------------------------
def whiten_ai(img, prompt_intensity="natural"):
    # 1) mērogs, lai ātrāk (SD labāk strādā 512–768px garajā malā)
    h, w = img.shape[:2]
    scale = 768 / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()

    # 2) ģenerē masku ar mediapipe (muti → mutes iekšpuse)
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            res = face_mesh.process(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                # Ja nav sejas – fallback uz fast
                return whiten_fast(img)

            sh, sw = img_small.shape[:2]
            mask = np.zeros((sh, sw), dtype=np.uint8)

            # Iekšējās lūpas/teeth tuvā zona:
            inner_idx = list(range(78, 88)) + list(range(308, 318))
            pts = []
            for lm in res.multi_face_landmarks[0].landmark:
                pts.append((int(lm.x * sw), int(lm.y * sh)))
            inner = np.array([pts[i] for i in inner_idx], dtype=np.int32)

            cv2.fillPoly(mask, [inner], 255)
            mask = cv2.GaussianBlur(mask, (21,21), 10)

    except Exception:
        # Ja mediapipe nav vai “misēklis” – fallback uz fast
        return whiten_fast(img)

    # 3) sagatavo SD inpainting ievadi
    from PIL import Image
    from diffusers import StableDiffusionInpaintPipeline
    import torch

    # Uz augšu uz SD izmēru
    if scale < 1.0:
        input_sd = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        input_sd = img.copy()

    # Maskai jābūt RGB ar baltu zonu (kur jālabo) un melnu citur
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    image_pil = Image.fromarray(cv2.cvtColor(input_sd, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_rgb).convert("RGB")

    # Prompt – izvēle pēc intensitātes
    if prompt_intensity == "hollywood":
        prompt = "whiten teeth significantly but naturally, keep gums, lips, skin and lighting unchanged, photorealistic, high quality"
    elif prompt_intensity == "bright":
        prompt = "whiten teeth a bit more than natural, keep gums, lips, skin and lighting unchanged, photorealistic"
    else:
        prompt = "whiten teeth naturally, keep gums, lips, skin and lighting unchanged, photorealistic"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=7.0,
        num_inference_steps=30
    ).images[0]

    out_sd = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # 4) ja samazinājām – uzskalo atpakaļ uz oriģinālo izmēru
    if scale < 1.0:
        out_final = cv2.resize(out_sd, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        out_final = out_sd

    return out_final

# --------------------------
# ROUTES
# --------------------------
@app.route("/")
def home():
    return jsonify({"message": "Teeth Whitening API – fast & ai modes ready 😁"})

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return jsonify({"error": "Nav augšupielādēta bilde"}), 400

    mode = request.form.get("mode", "fast")  # fast | ai
    intensity = int(request.form.get("intensity", 25))  # fast režīmam (10–50)
    ai_level = request.form.get("ai_level", "natural")  # natural | bright | hollywood

    # ielasa attēlu
    f = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        f.save(tmp.name)
        img = cv2.imread(tmp.name)

    if img is None:
        return jsonify({"error": "Nevar nolasīt attēlu"}), 400

    try:
        if mode == "ai":
            out = whiten_ai(img, prompt_intensity=ai_level)
        else:
            out = whiten_fast(img, intensity=intensity)
        return _save_and_send(out, "whitened.jpg")
    except Exception as e:
        # Ja AI režīms krīt – fallback uz fast
        try:
            out = whiten_fast(img, intensity=intensity)
            return _save_and_send(out, "whitened_fallback.jpg")
        except Exception:
            return jsonify({"error": f"Apstrādes kļūda: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
