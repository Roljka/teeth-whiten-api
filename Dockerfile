FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY teeth_api.py .

ENV PORT=10000
EXPOSE 10000
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-t", "120", "-b", "0.0.0.0:10000", "teeth_api:app"]
