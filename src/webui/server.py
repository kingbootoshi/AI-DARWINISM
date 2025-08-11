"""
FastAPI server exposing a minimal real-time dashboard for GEPA runs.

Endpoints
---------
- GET /state: JSON snapshot for polling
- POST /rate: human-in-the-loop ratings for horror stories

Static UI is served from / (index.html) with simple JS polling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .state import state


app = FastAPI(title="Darwinism GEPA UI")

# Allow local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/state")
async def get_state() -> JSONResponse:
    """Return the current shared state."""
    return JSONResponse(content=state.to_dict())


@app.post("/rate")
async def rate_story(req: Request) -> JSONResponse:
    """Receive human ratings for a generated story (HITL).

    Expected JSON body:
    { "story": str, "scariness": int, "suspense": int, "originality": int, "clarity": int, "rule_two_sentences": int }
    """
    data: Dict[str, Any] = await req.json()
    story = (data.get("story") or "").strip()
    # Minimal validation; in production persist to a DB/CSV
    ratings = {
        k: int(data.get(k, 0))
        for k in ("scariness", "suspense", "originality", "clarity", "rule_two_sentences")
    }
    state.log_event("human_rating", {"story": story, **ratings})
    return JSONResponse({"ok": True})


@app.get("/")
async def index() -> HTMLResponse:
    """Serve a tiny dashboard page with polling JS."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def run(host: str = "127.0.0.1", port: int = 3000) -> None:
    """Convenience entry to run uvicorn programmatically."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=False)


