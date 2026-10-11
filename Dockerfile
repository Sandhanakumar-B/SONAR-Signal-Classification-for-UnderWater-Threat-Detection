# =============================================================================
# SONAR Signal Classification System — Production Dockerfile
# Underwater Threat Detection: Tactical Defense Dashboard & REST API
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="Sandhanakumar-B"
LABEL description="Tactical Sonar Signal Classification System - Web Dashboard & REST API"
LABEL version="1.0.0"

# Install curl for container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application data, trained models, source code, static assets and templates
COPY data/ ./data/
COPY models/ ./models/
COPY results/ ./results/
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY app.py .
COPY test_environment.py .
COPY verify_system.py .

# Create non-root user for container security hardening (principle of least privilege)
RUN groupadd -r sonarapp && useradd -r -g sonarapp -d /app -s /sbin/nologin sonarapp && \
    chown -R sonarapp:sonarapp /app

USER sonarapp

# Expose Flask REST API & Web Dashboard port
EXPOSE 5000

ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production

# Docker healthcheck testing the /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Entrypoint: start Tactical Sonar Classification web application
ENTRYPOINT ["python", "app.py"]
