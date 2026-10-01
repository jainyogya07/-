import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "AI-Driven Extreme Weather Anomaly Tracking API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/weather"
    
    # Storage and Caching
    DATA_DIR: Path = BASE_DIR / "data" / "raw"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"
    
    # News API Key (NewsAPI.org key if provided, otherwise GDELT / ReliefWeb fallback)
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    NEWS_PROVIDER_FALLBACK: bool = os.getenv("NEWS_PROVIDER_FALLBACK", "True").lower() in ("true", "1")
    
    # Coordinates for key meteorological regions in India / South Asia
    REGIONS: dict = {
        "bay_of_bengal": {"name": "Bay of Bengal (Cyclone Basin)", "bbox": [80.0, 8.0, 95.0, 22.0]},
        "arabian_sea": {"name": "Arabian Sea Basin", "bbox": [65.0, 10.0, 75.0, 24.0]},
        "north_india": {"name": "North India (Heatwave/Coldwave zone)", "bbox": [72.0, 25.0, 85.0, 35.0]},
        "coastal_odisha": {"name": "Odisha Coastal Zone", "bbox": [83.0, 18.0, 87.5, 22.5]},
        "western_ghats": {"name": "Western Ghats (Orographic Rain)", "bbox": [73.0, 8.0, 77.0, 20.0]}
    }

    def init_dirs(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_dirs()
