FROM python:3.11-slim-bullseye

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /home/aicrowd
COPY . .
