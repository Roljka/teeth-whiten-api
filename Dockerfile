FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY teeth_api.py .

ENV PORT=10000
CMD ["gunicorn","teeth_api:app","--bind","0.0.0.0:10000","--workers","1","--threads","8","--timeout","120"]
