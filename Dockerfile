FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY teeth_api.py .

# Gunicorn ar gthread – pietiek 1 workeram, vari pacelt skaitu, ja vajag
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:10000", "teeth_api:app"]
