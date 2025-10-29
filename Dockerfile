FROM python:3.10-slim

# Sistēmas atkarības OpenCV (headless) darbībai
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Prasības
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# App
COPY teeth_api.py .

# Gunicorn start
ENV PORT=10000
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-t", "180", "-b", "0.0.0.0:10000", "teeth_api:app"]
