FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy configuration
COPY pyproject.toml .

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy source code
COPY src/ src/
COPY main copy.ipynb . 

# Create data and artifacts directories
RUN mkdir -p data/raw artifacts

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080 
# Configurable port (though not used by the script directly, good for platform compliance)

# Default command: Run the scheduler
CMD ["python", "src/scheduler.py"]
