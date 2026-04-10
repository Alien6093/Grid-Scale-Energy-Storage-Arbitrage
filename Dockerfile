FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install uv and install dependencies globally (system-wide)
RUN pip install uv && \
    uv pip install --system .

# Expose the server port
EXPOSE 7860

# Start the server using system python
CMD ["python", "-m", "server.app"]
