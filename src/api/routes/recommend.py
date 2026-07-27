from fastapi import APIRouter, HTTPException

from src.api.inference import recommender_service
from src.api.schemas import RecommendRequest, RecommendResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> RecommendResponse:
    if not recommender_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não disponível em Production no MLflow Registry.",
        )
    try:
        items = recommender_service.recommend(payload.user_id, payload.k)
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecommendResponse(user_id=payload.user_id, recommendations=items)
