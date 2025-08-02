# Simple Dockerfile for Congressional Trading Intelligence System
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the simple app
COPY simple_app.py .
COPY analysis_output/ ./analysis_output/

# Create a robust start script that handles PORT properly
RUN echo '#!/bin/bash\nPORT=${PORT:-5000}\necho "🚂 Congressional Trading Intelligence System"\necho "Starting gunicorn on 0.0.0.0:$PORT"\necho "Environment: PORT=$PORT"\nexec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --access-logfile - --error-logfile - simple_app:app' > start.sh && chmod +x start.sh

# Don't set default PORT - let Railway set it
EXPOSE 5000

CMD ["./start.sh"]