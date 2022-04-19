# syntax=docker/dockerfile:1

FROM python:3.8.9

COPY / /app/

WORKDIR /app

RUN wget https://tf.novaal.de/barcelona/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl

ENV DOCKERMODE=1
RUN pip3 install -e .
RUN pip install --ignore-installed --upgrade tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
RUN ls > aa.txt