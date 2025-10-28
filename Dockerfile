FROM python:3.10-slim

WORKDIR /app
COPY . .

# OpenCV vajadzīgās sistēmas bibliotēkas
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:$PORT teeth_api:app
