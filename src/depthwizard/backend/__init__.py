"""backend module -- API scaffold, NOT a working service in M1.

M1's frontend viewer does not talk to this API; it loads a static scene
file produced by scripts/export_sample_scene.py (see that script and
frontend/README.md). This module is a minimal FastAPI skeleton for a
future milestone's live pipeline endpoint.

Status: UNVERIFIED in the development session that created it. `fastapi`
and `uvicorn` are not installed there and could not be installed (no
network access to PyPI in that session) -- so `app.py` has never actually
been run or import-tested. Treat it as source code to review, not as a
working service, until someone runs it with the `backend` extra installed.
"""
