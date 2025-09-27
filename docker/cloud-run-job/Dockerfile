# syntax=docker/dockerfile:1.7

FROM python:3.11.7-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off

WORKDIR /tmp/build
COPY docker/cloud-run-job/requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip wheel --wheel-dir /tmp/wheels -r requirements.txt

FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/output

WORKDIR /app

COPY --from=builder /tmp/wheels /wheels
COPY docker/cloud-run-job/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src ./src
COPY docker/cloud-run-job/cloud_run_job.py ./cloud_run_job.py

RUN groupadd --system app && useradd --system --create-home --gid app appuser \
    && mkdir -p ${OUTPUT_DIR} \
    && chown -R appuser:app /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["python", "cloud_run_job.py", "--healthcheck"]

ENTRYPOINT ["python", "cloud_run_job.py"]
CMD ["run"]
