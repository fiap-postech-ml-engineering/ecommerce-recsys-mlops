from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.inference import recommender_service
from src.api.middleware import register_observability_middleware
from src.api.routes.recommend import router as recommend_router
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    recommender_service.load()
    yield


settings = get_settings()

app = FastAPI(
    title="E-commerce RecSys API",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)

register_observability_middleware(app)
app.include_router(recommend_router)


@app.get("/")
def home():
    return {
        "service": "E-commerce RecSys API",
        "version": settings.MODEL_VERSION,
        "env": settings.APP_ENV,
        "status": "ok",
    }
