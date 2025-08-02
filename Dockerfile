# Simple Dockerfile for Congressional Trading Intelligence System
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the simple app
COPY simple_app.py .
COPY analysis_output/ ./analysis_output/

# Don't create complex start scripts - just use Python
EXPOSE 5000

CMD ["python", "simple_app.py"]