"""AWS Lambda handler using Mangum ASGI adapter."""
from __future__ import annotations

from mangum import Mangum

from .app import app

handler = Mangum(app, lifespan="off")
