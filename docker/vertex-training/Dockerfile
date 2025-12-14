# syntax=docker/dockerfile:1.7

FROM python:3.11.7-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off

WORKDIR /tmp/build
COPY docker/vertex-training/requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip wheel --wheel-dir /tmp/wheels -r requirements.txt

FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /tmp/wheels /wheels
COPY docker/vertex-training/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY docker/vertex-training/train.py ./train.py
COPY docker/vertex-training/predict.py ./predict.py

RUN groupadd --system trainer && useradd --system --create-home --gid trainer trainer \
    && chown -R trainer:trainer /app

USER trainer

# Default to train.py, but can run predict.py via --entrypoint or args
ENTRYPOINT ["python"]
CMD ["train.py"]

