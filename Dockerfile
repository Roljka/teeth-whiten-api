FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1 \
    MEDIAPIPE_GLOBAL_DISABLE_GPU=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY teeth_api.py .

EXPOSE 10000
# + lielāks timeout smagajam pirmajam pieprasījumam
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "2", \
     "--graceful-timeout", "60", "--timeout", "240", \
     "-b", "0.0.0.0:10000", "teeth_api:app"]
