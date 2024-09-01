FROM python:3.11-bookworm

RUN apt-get update -qq && apt-get install -y -qq libboost-all-dev cmake

WORKDIR /qlbm

COPY Makefile Makefile
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN make install-cpu

COPY qlbm qlbm
COPY test test

ENTRYPOINT /bin/bash