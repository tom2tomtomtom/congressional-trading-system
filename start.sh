#!/bin/bash
# Railway start script for Congressional Trading Intelligence System

# Use Railway's PORT environment variable, fallback to 5000
PORT=${PORT:-5000}

echo "🚂 Starting Congressional Trading Intelligence System on port $PORT"

# Start gunicorn with the specified port
exec gunicorn --bind 0.0.0.0:$PORT simple_app:app