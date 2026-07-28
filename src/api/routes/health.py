from fastapi import APIRouter

from src.api.inference import recommender_service

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "api_status": "operacional",
        "modelo_carregado": recommender_service.is_ready,
    }
