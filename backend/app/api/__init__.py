from app.api.weather_routes import router as weather_router
from app.api.news_routes import router as news_router

__all__ = ["weather_router", "news_router"]
