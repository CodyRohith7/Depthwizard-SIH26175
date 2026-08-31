"""Minimal FastAPI skeleton for a future live pipeline endpoint.

NOT part of M1's definition of done, and NOT verified to run (see the
backend package docstring). Provided now only so the /backend folder in
the approved architecture is not empty, and so the eventual live-endpoint
work has a starting shape to build from.

To try it (on a machine with real network access):
    pip install -e ".[backend]"
    uvicorn depthwizard.backend.app:app --reload
"""
from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastapi is not installed. Install the `backend` extra: "
        "pip install -e '.[backend]'"
    ) from exc

app = FastAPI(title="DepthWizard API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "service": "depthwizard-backend",
        "status": "scaffold-only",
        "note": "No pipeline endpoints implemented yet -- see Milestone M2+.",
    }
