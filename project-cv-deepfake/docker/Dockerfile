# dfdetect serving image — torch-free (ONNX Runtime only), so this is much smaller
# and faster to build than the training environment. Bakes code + config; mount the
# exported model as a volume (-v $(pwd)/data:/app/data) or bake it via COPY.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
COPY conf/ conf/
RUN pip install --no-cache-dir -e .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Mount the exported model:  docker run -v $(pwd)/data:/app/data -p 8000:8000 dfdetect
CMD ["uvicorn", "dfdetect.serve:app", "--host", "0.0.0.0", "--port", "8000"]
