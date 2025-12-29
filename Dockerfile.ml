FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy ML core code
COPY miq/ ./miq/
COPY machine_iq_core.py .
COPY ml_service.py .

# Create models directory
RUN mkdir -p /app/models

# Expose API port
EXPOSE 5000

# Run the ML service
CMD ["python", "ml_service.py"]
