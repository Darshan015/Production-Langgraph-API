FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Create non-root user for security
RUN useradd --create-home appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check (Note: python:3.12-slim doesn't have curl by default, we use python instead)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn directly (since we aren't using uv anymore)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]