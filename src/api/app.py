from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import register_observability_middleware
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    yield


settings = get_settings()

app = FastAPI(
    title="E-commerce RecSys API",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)

register_observability_middleware(app)


@app.get("/")
def home():
    return {
        "service": "E-commerce RecSys API",
        "version": settings.MODEL_VERSION,
        "env": settings.APP_ENV,
        "status": "ok",
    }
