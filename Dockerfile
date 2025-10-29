FROM python:3.10-slim

# OpenCV un diffusers vajadzīgās sistēmas bibliotēkas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Kešu mapes (lai nesprāgst diska kvota un paātrina startu)
ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_CACHE=/tmp/hf
ENV MPLCONFIGDIR=/tmp/mpl

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Izvēles solis: pre-cache modeli build laikā (ātrākam pirmajam startam)
# Ja Free plāns/metru limits, vari izkomentēt šo rindu.
RUN python - << 'PY'
from diffusers import StableDiffusionInpaintPipeline
StableDiffusionInpaintPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_inpainting")
PY

# Startēšana
CMD gunicorn -w 1 -k gthread --threads 4 --timeout 300 --bind 0.0.0.0:10000 teeth_api:app
