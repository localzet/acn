FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip \
    && pip install mlflow==2.19.0 psycopg2-binary==2.9.10 boto3==1.35.99

EXPOSE 5000

