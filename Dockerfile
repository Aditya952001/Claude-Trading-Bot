# ==============================================================================
# Dockerfile for Hugging Face Spaces (CPU Basic - Free Tier)
# ==============================================================================
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user:user requirements.txt $HOME/app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio

# Copy application files
COPY --chown=user:user . $HOME/app/

# Ensure logs and data directories exist
RUN mkdir -p logs data

# Expose standard Hugging Face port
EXPOSE 7860

# Start app.py (which runs the web dashboard & trading bot concurrently)
CMD ["python", "app.py"]
