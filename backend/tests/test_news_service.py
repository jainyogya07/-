"""Tests for Live Extreme Weather News Service and provider fallbacks."""

import pytest
from app.news.news_service import WeatherNewsService


def test_mock_fallback_news():
    service = WeatherNewsService()
    articles = service.get_mock_fallback_news("cyclone", "Odisha")

    assert len(articles) >= 1
    assert "Cyclone" in articles[0]["title"]
    assert "Odisha" in articles[0]["title"]
    assert "url" in articles[0]
    assert "published_at" in articles[0]


def test_get_live_disaster_news_execution():
    service = WeatherNewsService()
    articles = service.get_live_disaster_news(hazard="heatwave", region="Rajasthan", limit=3)

    assert isinstance(articles, list)
    assert len(articles) >= 1
    article = articles[0]
    assert "title" in article
    assert "source" in article
    assert "url" in article


def test_news_api_key_update():
    service = WeatherNewsService(api_key="test_key_123")
    assert service.api_key == "test_key_123"
    service.api_key = "updated_key_456"
    assert service.api_key == "updated_key_456"
