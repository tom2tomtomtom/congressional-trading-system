FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-production.txt .
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p logs models/phase2 analysis/network

# Environment
ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000
CMD ["python", "src/api/app.py"]