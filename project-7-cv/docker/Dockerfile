# Overwatch detection service. Bakes code + config; mount trained weights at
# data/runs/finetune/weights/best.pt (else it falls back to pretrained yolov8n.pt,
# which ultralytics downloads on first run).
FROM python:3.12-slim

# OpenCV / ultralytics need these system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
COPY conf/ conf/
RUN pip install --no-cache-dir -e .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Mount trained weights:  docker run -v $(pwd)/data:/app/data -p 8000:8000 overwatch
CMD ["uvicorn", "overwatch.serve:app", "--host", "0.0.0.0", "--port", "8000"]
