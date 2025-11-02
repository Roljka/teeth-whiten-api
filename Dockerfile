FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install -r requirements.txt

COPY teeth_api.py .

EXPOSE 10000
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:10000", "teeth_api:app"]
