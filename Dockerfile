FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY teeth_api.py .

# Render parasti iedod PORT=10000, bet lai ir elastīgi:
ENV PORT=10000

CMD ["sh", "-c", "gunicorn teeth_api:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 180"]
