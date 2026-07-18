# Serving image for the Spine NIDS detector.
# The promoted model is baked in (run scripts/export_model.py first), so the
# container serves with no dependency on an MLflow server.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Deps first for layer caching.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# App source + config + the exported model.
COPY pyproject.toml .
COPY src ./src
COPY conf ./conf
COPY serving/model ./serving/model
RUN pip install -e . --no-deps

ENV SPINE_MODEL_PATH=/app/serving/model
EXPOSE 8000

# Simple container healthcheck against the /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "spine.serve:app", "--host", "0.0.0.0", "--port", "8000"]
