FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux asciinema git ripgrep python3-pytest python3-requests \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY app.py /workspace/app.py
