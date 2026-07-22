from pydantic import BaseModel

from src.config import get_settings


class RecommendRequest(BaseModel):
    user_id: int
    k: int = get_settings().RECOMMENDATION_K


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[int]