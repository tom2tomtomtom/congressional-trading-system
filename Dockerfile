# Simple Dockerfile for Congressional Trading Intelligence System
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the simple app
COPY simple_app.py .
COPY analysis_output/ ./analysis_output/

# Create a simple start script
RUN echo '#!/bin/bash\nexec gunicorn --bind 0.0.0.0:$PORT simple_app:app' > start.sh && chmod +x start.sh

# Use Railway's PORT environment variable
ENV PORT=5000

EXPOSE $PORT

CMD ["./start.sh"]