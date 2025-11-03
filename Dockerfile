FROM python:3.10-slim

# Mazās runtime atkarības mediapipe/opencv (bez libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tavs API fails:
COPY teeth_api.py .

ENV PORT=10000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 10000

# Vienkāršs healthcheck (nav obligāts, vari mest ārā)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD \
  curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Startēšana ar Gunicorn (gthread ir ideāli šim API)
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:10000", "teeth_api:app"]
