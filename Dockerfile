FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --no-cache-dir

RUN uv pip install --system --no-cache \
    "torch>=2.3.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# Tell huggingface_hub where to find model weights.
# The models/ directory is volume-mounted at runtime (see docker-compose.yml)
# so weights are never baked into the image — keeping image size manageable.
ENV HF_HUB_CACHE=/app/models
# Ensure `from app.xxx import` resolves correctly when running from /app
ENV PYTHONPATH=/app

COPY app/ ./app/
COPY corpus/ ./corpus/

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.runOnSave=true", \
     "--server.fileWatcherType=poll"]
