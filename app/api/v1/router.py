"""API v1 router aggregation."""
from fastapi import APIRouter

from app.api.v1.endpoints import health, predict

v1_router = APIRouter()
v1_router.include_router(health.router, tags=["Health"])
v1_router.include_router(predict.router, tags=["Sentiment Analysis"])
