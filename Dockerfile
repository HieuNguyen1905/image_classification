FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Upgrade pip and install uv
RUN pip install --upgrade pip && \
    pip install uv

# Install dependencies before code to cache layer
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

# Copy remaining code (configs, weights, src)
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]