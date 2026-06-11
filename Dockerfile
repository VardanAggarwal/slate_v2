FROM python:3.12-slim

# curl for the compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the sentence tokenizer so first boot never depends on nltk's CDN.
# (The embedding model is NOT baked — it lands in the hf-cache volume on
# first use, keeping the image smaller and the cache reusable across builds.)
RUN python -m nltk.downloader punkt_tab

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
