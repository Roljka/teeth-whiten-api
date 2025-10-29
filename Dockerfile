FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# (A) Minimālās sistēmas bibl.: libgl1 un glib – aizver “libGL.so.1” caurumu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY teeth_api.py .

# Gunicorn
ENV PORT=10000
CMD ["gunicorn","-w","2","-k","gthread","--threads","4","--timeout","120","-b","0.0.0.0:10000","teeth_api:app"]
