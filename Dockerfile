# Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables:
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures standard output/error are not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /gymnasium-http-api

# Install system dependencies needed for Gymnasium
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies from the file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository code into the container
COPY app app

# Expose the port the FastAPI server will run on
EXPOSE 5000

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
