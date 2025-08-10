FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/
COPY models/iris_rf_model.joblib /app/models/

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV MODEL_PATH=/app/models/iris_rf_model.joblib
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000  

# Expose port
EXPOSE 5001

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "app:app"]
