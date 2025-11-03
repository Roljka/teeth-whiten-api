FROM python:3.10-slim

# Sistēmas bibliotēkas, ko nereti prasa OpenCV/MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates curl \
    libgl1 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 # ja kāds transitive atnes "opencv-python", brutāli izmetam:
 && pip uninstall -y opencv-python opencv-contrib-python || true \
 # early-fail, ja kaut kas trūkst:
 && python - <<'PY'
import cv2, sys
print("OpenCV:", cv2.__version__)
PY

COPY teeth_api.py .

ENV PORT=10000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_LOG_LEVEL=ERROR

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD \
  curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["gunicorn","-w","1","-k","gthread","--threads","4","--timeout","120","-b","0.0.0.0:10000","teeth_api:app"]
