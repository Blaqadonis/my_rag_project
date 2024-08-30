# Use a base image with Python 3.10
FROM python:3.10-slim

# Install system dependencies required for pdf2image
RUN apt-get update && \
    apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py"]
