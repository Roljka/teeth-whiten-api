FROM python:3.10-slim

# Sistēmas bibliotēkas OpenCV/Diffusers vajadzībām
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Ja gribi: pip jaunākais
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cache uz /tmp, lai nesprāgst diska kvotas
ENV TRANSFORMERS_CACHE=/tmp/hf
ENV HF_HOME=/tmp/hf

# Noklusētā palaišana (pielāgo Render “Start Command”, ja vēlies)
CMD gunicorn -w 1 -k gthread --threads 4 --timeout 180 --bind 0.0.0.0:10000 teeth_api:app
# Kešot modeli build laikā
RUN python -c "from diffusers import StableDiffusionInpaintPipeline; StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting')"
