# Velos backend image: serves the API (default) or runs the worker (command override).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VELOS_STORAGE_DIR=/data
WORKDIR /app
COPY pyproject.toml ./
COPY velos_recon ./velos_recon
COPY velos_backend ./velos_backend
RUN pip install --no-cache-dir . && mkdir -p /data
EXPOSE 8000
CMD ["uvicorn", "velos_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
