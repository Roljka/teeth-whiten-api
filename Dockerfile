FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY teeth_api.py .

ENV PORT=10000
EXPOSE 10000

CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:10000", "teeth_api:app"]
