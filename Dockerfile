FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -e .

# Application code
COPY pymem/ pymem/
COPY alembic/ alembic/

# Create data directory
RUN mkdir -p /app/data/kuzu

EXPOSE 8001

CMD ["uvicorn", "pymem.main:app", "--host", "0.0.0.0", "--port", "8001"]
