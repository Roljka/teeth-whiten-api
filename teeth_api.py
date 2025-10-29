from diffusers import StableDiffusionInpaintPipeline
import torch, cv2, numpy as np
from flask import Flask, request, send_file
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_ID = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # mazāk RAM
    revision="fp16"
)
pipe.to("cpu")  # render bez GPU
pipe.enable_attention_slicing()  # vēl viens RAM ietaupījums

def whiten_teeth(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 180, 255)  # gaišās zonas (zobi)
    mask = cv2.dilate(mask, None, iterations=2)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l = cv2.add(l, 25)
    img_lab = cv2.merge((l, a, b))
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    img_bgr[mask == 0] = img_bgr[mask == 0]
    return img_bgr

@app.route("/whiten", methods=["POST"])
def whiten():
    if "file" not in request.files:
        return {"error": "no file"}, 400

    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    whitened = whiten_teeth(img)

    _, buffer = cv2.imencode(".jpg", whitened)
    return send_file(BytesIO(buffer), mimetype="image/jpeg")

@app.route("/")
def home():
    return "Teeth Whitening API v2 – online 😁"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
