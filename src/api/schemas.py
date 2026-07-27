from pydantic import BaseModel, Field

from src.config import get_settings


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default_factory=lambda: get_settings().RECOMMENDATION_K, gt=0)


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[int]
