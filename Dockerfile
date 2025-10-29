FROM python:3.10-slim

# Instalē sistēmas atkarības priekš OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start komanda
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", "teeth_api:app"]
