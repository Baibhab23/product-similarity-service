# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and dataset into the container
COPY app ./app
COPY data ./data

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Make port 8000 available to the world outside this container
EXPOSE 8000

ENV NAME=ProductSimilarityApp \
    PYTHONUNBUFFERED=1 \
    SIMILARITY_BACKEND=brute \
    IMAGE_ENABLED=false \
    CHROMA_PATH=/app/data/chroma \
    CHROMA_COLLECTION=product_images \
    RRF_K=60

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Run the FastAPI app when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
