FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8001

# CMD /opt/conda/envs/ldm/bin/ray stop && /opt/conda/envs/ldm/bin/ray start --head --disable-usage-stats && 

# # Run the application
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
# Use a shell form CMD to run multiple commands
CMD /opt/conda/bin/ray stop && \
    /opt/conda/bin/ray start --head --disable-usage-stats && \
    uvicorn app:app --host 0.0.0.0 --port 8001