"""Run the Mini Photoshop FastAPI web app."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=80, reload=True)
