# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files to the container
COPY app.py .
COPY config.yaml .

# Expose the port Gradio will use
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]