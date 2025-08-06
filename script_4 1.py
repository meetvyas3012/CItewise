# Create a simplified Dockerfile specifically for Streamlit deployment
streamlit_dockerfile = '''# Streamlit Deployment Dockerfile for Multi-Document Research Assistant
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for document processing
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    tesseract-ocr-eng \\
    poppler-utils \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application files
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data/documents data/embeddings data/models data/temp logs \\
    && chmod 777 data data/documents data/embeddings data/models data/temp logs

# Create .streamlit directory and copy config
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port
EXPOSE 8501

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Run the Streamlit application
CMD ["streamlit", "run", "deploy_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
'''

with open('Dockerfile.streamlit', 'w') as f:
    f.write(streamlit_dockerfile)

# Create docker-compose for Streamlit deployment
streamlit_compose = '''version: '3.8'

services:
  multi-doc-rag:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: multi-doc-rag-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

networks:
  default:
    name: multi-doc-rag-network
'''

with open('docker-compose.streamlit.yml', 'w') as f:
    f.write(streamlit_compose)

print("✅ Streamlit Docker configuration created")