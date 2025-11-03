FROM python:3.10-slim

# Sistēmas libi, ko prasa OpenCV/Mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# 1) Uzinstalējam requirements
# 2) Ja kaut kas atvelk "opencv-python"/"opencv-contrib-python" → izmetam
# 3) Piespiežam tieši headless buildu (bez deps), lai vienmēr ir cv2
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y opencv-python opencv-contrib-python || true \
 && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.8.1.78

COPY teeth_api.py .

ENV PORT=10000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 10000

# (nav obligāts, bet ērti Renderam)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD \
  curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["gunicorn","-w","1","-k","gthread","--threads","4","--timeout","120","-b","0.0.0.0:10000","teeth_api:app"]
