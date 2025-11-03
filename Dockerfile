FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Sistēmas bibliotēkas, kas izbeidz "libGL.so.1" bļāvienus
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Ļoti skaidri: NOMETAM jebkuru opencv (ja atnācis tranzitīvi),
# un uzliekam tikai headless
RUN pip install --upgrade pip && \
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true && \
    pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'\nimport cv2; print("OpenCV OK:", cv2.__version__)\nPY

COPY teeth_api.py .

EXPOSE 10000
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:10000", "teeth_api:app"]
