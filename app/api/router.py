"""Root API router."""
from fastapi import APIRouter

from app.api.v1.router import v1_router

api_router = APIRouter()
# Include v1 endpoints both at root level (for POST /predict and GET /health direct access)
# and prefixed under /api/v1 for versioned clients.
api_router.include_router(v1_router)
api_router.include_router(v1_router, prefix="/api/v1")
