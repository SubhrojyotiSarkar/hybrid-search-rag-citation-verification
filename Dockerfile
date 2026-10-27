# One image for both services; docker-compose.yml picks the command (backend or frontend).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face models (embedding, tokenizer, cross-encoder) are cached in a volume, so
    # they are downloaded once, not on every container start.
    HF_HOME=/models

WORKDIR /app

# PyMuPDF, faiss-cpu and torch ship self-contained wheels: no system packages needed.
# requirements.txt points pip at the CPU-only PyTorch index (no multi-GB CUDA build).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY tests ./tests
COPY pytest.ini .

# Run as an unprivileged user. data/ and evaluation/ are mounted from the host.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/evaluation /models \
    && chown -R app:app /app /models
USER app

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
