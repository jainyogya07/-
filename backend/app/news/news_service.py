"""
Live Extreme Weather Disaster News Integration.
Integrates live news media feeds (NewsAPI, GDELT Project 2.0, ReliefWeb)
with spatio-temporal extreme weather anomaly detection layers.
"""

from typing import Any
import os
import logging
import requests
from app.config import settings

logger = logging.getLogger(__name__)


class WeatherNewsService:
    """
    Fetches real-time disaster and extreme weather news bulletins.
    Correlates live media reports with detected weather anomaly footprints.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.NEWS_API_KEY
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WeatherAnomalyTracker/1.0 (MoES Hackathon)"})

    def fetch_news_api(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Queries NewsAPI.org v2 everything endpoint."""
        if not self.api_key:
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": limit,
            "language": "en",
            "apiKey": self.api_key,
        }

        try:
            resp = self.session.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                articles = []
                for item in data.get("articles", []):
                    articles.append({
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "source": item.get("source", {}).get("name", "News Outlet"),
                        "url": item.get("url"),
                        "published_at": item.get("publishedAt"),
                        "provider": "NewsAPI",
                    })
                return articles
            else:
                logger.warning("NewsAPI returned status %s: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("NewsAPI request failed: %s", e)

        return []

    def fetch_gdelt_news(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Queries GDELT 2.0 Doc API (Global Database of Events, Language, and Tone).
        Requires no API key, updates every 15 minutes globally.
        """
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": f"{query} sourcelang:eng",
            "mode": "artlist",
            "maxrecords": str(limit),
            "format": "json",
            "sort": "DateDesc",
        }

        try:
            resp = self.session.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                articles = []
                for item in data.get("articles", []):
                    articles.append({
                        "title": item.get("title"),
                        "description": f"Reported from {item.get('domain', 'News Source')}",
                        "source": item.get("domain", "GDELT Global News"),
                        "url": item.get("url"),
                        "published_at": item.get("seendate"),
                        "provider": "GDELT Project",
                    })
                return articles
        except Exception as e:
            logger.warning("GDELT API request failed: %s", e)

        return []

    def fetch_reliefweb_news(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Queries UN OCHA ReliefWeb API for disaster bulletins."""
        url = "https://api.reliefweb.int/v1/reports"
        params = {
            "appname": "MoES_ExtremeWeather_AI",
            "query[value]": f"{query} disaster",
            "limit": limit,
            "sort[]": "date:desc",
        }

        try:
            resp = self.session.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                articles = []
                for item in data.get("data", []):
                    fields = item.get("fields", {})
                    articles.append({
                        "title": fields.get("title"),
                        "description": fields.get("body-summary", "Humanitarian weather alert."),
                        "source": "UN OCHA ReliefWeb",
                        "url": fields.get("url"),
                        "published_at": fields.get("date", {}).get("created"),
                        "provider": "ReliefWeb",
                    })
                return articles
        except Exception as e:
            logger.warning("ReliefWeb API request failed: %s", e)

        return []

    def get_mock_fallback_news(self, hazard: str, region: str) -> list[dict[str, Any]]:
        """Guarantees resilient responses during hackathon presentations or offline runs."""
        return [
            {
                "title": f"IMD Issues Red Alert for Severe {hazard.replace('_', ' ').title()} in {region.replace('_', ' ').title()}",
                "description": f"National Disaster Response Force (NDRF) teams pre-deployed across vulnerable districts following extreme medium-range forecast models.",
                "source": "India Meteorological Department / MoES Bulletin",
                "url": "https://mausam.imd.gov.in",
                "published_at": "2026-10-01T12:00:00Z",
                "provider": "MoES Emergency Fallback Feed",
            },
            {
                "title": f"State Disaster Management Authority Activates Warning Protocols for {region.title()}",
                "description": "District collectors directed to prepare evacuation shelters and secure fishing boat operations over the next 72 to 120 hours.",
                "source": "Disaster Management Authority",
                "url": "https://ndma.gov.in",
                "published_at": "2026-10-01T10:30:00Z",
                "provider": "MoES Emergency Fallback Feed",
            },
        ]

    def get_live_disaster_news(
        self,
        hazard: str = "cyclone",
        region: str = "India",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Unified news fetcher:
        1. Tries NewsAPI if API key is configured.
        2. Tries GDELT Project live feed.
        3. Tries ReliefWeb disaster feed.
        4. Falls back to realistic emergency mock feed if network/API fails.
        """
        clean_hazard = hazard.replace("_", " ")
        clean_region = region.replace("_", " ")
        query = f"{clean_hazard} {clean_region} weather"

        # 1. Try NewsAPI if key is present
        if self.api_key:
            results = self.fetch_news_api(query, limit=limit)
            if results:
                return results

        # 2. Try GDELT (Free, No Key)
        results = self.fetch_gdelt_news(query, limit=limit)
        if results:
            return results

        # 3. Try ReliefWeb
        results = self.fetch_reliefweb_news(query, limit=limit)
        if results:
            return results

        # 4. Resilience Fallback
        return self.get_mock_fallback_news(clean_hazard, clean_region)


news_service = WeatherNewsService()
