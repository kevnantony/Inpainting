# FROM python:3.11
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8002"]
