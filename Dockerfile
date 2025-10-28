FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:$PORT teeth_api:app
