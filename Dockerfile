FROM python:3.12-slim

# curl for the compose healthcheck; node + Claude Code CLI for the
# subscription-billed claude-cli LLM provider (CLAUDE_CODE_OAUTH_TOKEN)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Server image embeds via HF Inference API (HF_TOKEN) — no torch. Run with
# requirements.txt instead only if you need local embeddings in a container.
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Bake the sentence tokenizer so first boot never depends on nltk's CDN.
RUN python -m nltk.downloader punkt_tab

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
