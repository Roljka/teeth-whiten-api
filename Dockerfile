FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENV PYTORCH_JIT=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

CMD ["gunicorn", "-b", "0.0.0.0:10000", "-k", "gthread", "--threads", "2", "teeth_api:app"]
