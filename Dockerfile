FROM debian:bullseye-slim

# Instalē Python un nepieciešamās sistēmas atkarības
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Start komanda
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", "teeth_api:app"]
