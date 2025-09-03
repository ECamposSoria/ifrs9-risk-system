ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION} as wheels
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip wheel setuptools \
 && mkdir -p /wheels \
 && pip wheel --wheel-dir=/wheels -r requirements.txt \
 && pip wheel --wheel-dir=/wheels fastapi==0.110.0 uvicorn[standard]==0.29.0 prometheus-client==0.20.0

FROM python:${PYTHON_VERSION}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 AGENT_NAME=integrator
WORKDIR /app
COPY --from=wheels /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels fastapi uvicorn[standard] prometheus-client
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt || true
COPY . /app
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -fsS http://localhost:${PORT}/healthz || exit 1
CMD ["bash", "-lc", "uvicorn src.agents.runtime_server:app --host 0.0.0.0 --port ${PORT}"]

