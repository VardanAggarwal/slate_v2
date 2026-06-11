"""Thin FastAPI app mounting /mcp + /health."""
from fastapi import FastAPI

from mcp_server import mcp

mcp_app = mcp.http_app(path="/")
app = FastAPI(title="slate-engine", lifespan=mcp_app.lifespan)
app.mount("/mcp", mcp_app)


@app.get("/health")
def health() -> dict:
    from core import store
    counts = store.stats(store.connect())
    return {"status": "ok", "episodes": counts["episodes"],
            "concepts": counts["concepts"]}
