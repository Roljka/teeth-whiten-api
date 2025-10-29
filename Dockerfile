FROM python:3.10-bullseye

# Instalē OpenCV atkarības
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "-b", "0.0.0.0:10000", "teeth_api:app"]
