"""
Live Extreme Weather News API Endpoints.
Provides live weather disaster news articles correlated with active anomaly layers.
"""

from fastapi import APIRouter, Query, Body
from pydantic import BaseModel
from app.news.news_service import news_service
from app.config import settings

router = APIRouter(prefix="/weather/news", tags=["Live Extreme Weather News"])


class NewsKeyUpdateRequest(BaseModel):
    api_key: str


@router.get("")
def get_weather_news(
    hazard: str = Query("cyclone", description="Extreme weather hazard (e.g., cyclone, heatwave, flood, coldwave)"),
    region: str = Query("India", description="Target region or state (e.g., Odisha, West Bengal, Rajasthan, Delhi)"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of articles to return"),
):
    """
    Fetches live news articles and disaster bulletins related to the given weather topic and region.
    Integrates with NewsAPI, GDELT Project 2.0, and ReliefWeb.
    """
    articles = news_service.get_live_disaster_news(hazard=hazard, region=region, limit=limit)
    return {
        "hazard": hazard,
        "region": region,
        "articles_count": len(articles),
        "articles": articles,
        "active_provider": "NewsAPI" if news_service.api_key else "GDELT Project (Free Live Feed)",
    }


@router.get("/status")
def get_news_api_status():
    """
    Returns the status of the News API configuration.
    """
    has_key = bool(news_service.api_key and len(news_service.api_key.strip()) > 0)
    return {
        "news_api_key_configured": has_key,
        "active_primary_provider": "NewsAPI.org" if has_key else "GDELT Project 2.0 (Zero-config free live feed)",
        "fallback_enabled": settings.NEWS_PROVIDER_FALLBACK,
    }


@router.post("/key")
def update_news_api_key(payload: NewsKeyUpdateRequest):
    """
    Updates the NewsAPI.org API key at runtime without restarting the server.
    """
    news_service.api_key = payload.api_key.strip()
    return {
        "status": "success",
        "message": "NewsAPI key updated successfully",
        "has_key": bool(news_service.api_key),
    }
