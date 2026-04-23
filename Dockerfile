FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --upgrade pip && \
    pip install setuptools wheel

# Install dependencies before code to cache layer
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy remaining code (configs, weights, src)
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]