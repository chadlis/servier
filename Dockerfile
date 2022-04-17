# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

COPY / /app/

WORKDIR /app

RUN pip3 install -e .
RUN ls > aa.txt