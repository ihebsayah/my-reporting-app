FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir \
    fastapi uvicorn pydantic pydantic-settings sqlalchemy \
    scikit-learn joblib numpy requests

# Copy application source
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY docs/ ./docs/

# Create writable directories
RUN mkdir -p artifacts/models/rf_confidence artifacts/models/ner artifacts/feedback

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
