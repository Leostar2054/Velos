# Velos backend image: serves the API (default command) or runs the worker (command override).
# CPU image with ONNX Runtime for the ML refinement model. GPU worker (M4): build with
#   --build-arg ML_EXTRA=ml-gpu  on a CUDA base image (see docs/deployment.md).
FROM python:3.12-slim
ARG ML_EXTRA=ml
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VELOS_STORAGE_DIR=/data VELOS_MODEL_DIR=/models MPLCONFIGDIR=/tmp/mpl
WORKDIR /app
COPY pyproject.toml README.md ./
COPY velos_recon ./velos_recon
COPY velos_backend ./velos_backend
RUN pip install --no-cache-dir ".[${ML_EXTRA}]" \
 && useradd --system --uid 10001 --home /app velos \
 && mkdir -p /data /models && chown -R velos /data
USER velos
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "velos_backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
