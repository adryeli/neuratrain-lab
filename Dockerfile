FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app.py ./
COPY app_pages/ ./app_pages/
COPY src/ ./src/
COPY data/ ./data/
# Explicit filename, not the whole .streamlit/ directory: if a local
# .streamlit/secrets.toml is ever created, this line must not be able to
# copy it into the image, regardless of .dockerignore.
COPY .streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
