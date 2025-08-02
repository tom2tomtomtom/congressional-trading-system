FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements (simple version)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

EXPOSE $PORT
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT simple_app:app"]