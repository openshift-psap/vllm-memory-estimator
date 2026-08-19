"""FastAPI application for the vLLM memory estimator."""
from __future__ import annotations

from importlib.metadata import version

from .cpu_compat import patch as _patch_cpu_compat

_patch_cpu_compat()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .routes import router  # noqa: E402

app = FastAPI(
    title="vLLM Memory Estimator API",
    description="Estimate GPU memory requirements for serving models with vLLM.",
    version=version("vllm-memory-estimator"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def run() -> None:
    import os

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
