"""
Module 15: Market & Mandi Intelligence Engine.
Integrates APMC mandi modal prices, arrival volumes, production benchmarks,
and short-term price volatility forecasts driven by extreme weather shocks.
"""

from typing import Any


class MarketIntelligenceEngine:
    """
    Connects weather hazard footprints with wholesale agricultural markets,
    supply corridor disruptions, and price volatility forecasts.
    """

    MANDI_DATA = {
        "odisha": {
            "mandi_name": "Bhubaneswar APMC Agricultural Market",
            "district": "Khordha",
            "state": "Odisha",
            "crop": "Rice / Paddy (Common & Grade A)",
            "modal_price_inr_quintal": 2260.0,
            "min_price_inr_quintal": 2150.0,
            "max_price_inr_quintal": 2420.0,
            "msp_benchmark_inr": 2183.0,
            "daily_arrivals_tonnes": 420.0,
            "normal_seasonal_arrival_tonnes": 580.0,
            "price_trend_7d_pct": +5.4,
            "primary_transit_corridors": ["NH-16 (Coastal Highway)", "Bhubaneswar-Puri Link"],
        },
        "west_bengal": {
            "mandi_name": "Burdwan Central APMC Mandi",
            "district": "Purba Bardhaman",
            "state": "West Bengal",
            "crop": "Rice / Paddy (Swarna & Ratna)",
            "modal_price_inr_quintal": 2310.0,
            "min_price_inr_quintal": 2200.0,
            "max_price_inr_quintal": 2490.0,
            "msp_benchmark_inr": 2183.0,
            "daily_arrivals_tonnes": 650.0,
            "normal_seasonal_arrival_tonnes": 820.0,
            "price_trend_7d_pct": +4.8,
            "primary_transit_corridors": ["NH-19 (Grand Trunk)", "Durgapur Expressway"],
        },
        "punjab": {
            "mandi_name": "Khanna Grain Market (Asia's Largest)",
            "district": "Ludhiana",
            "state": "Punjab",
            "crop": "Wheat (PBW 550 / HD 2967)",
            "modal_price_inr_quintal": 2450.0,
            "min_price_inr_quintal": 2275.0,
            "max_price_inr_quintal": 2580.0,
            "msp_benchmark_inr": 2275.0,
            "daily_arrivals_tonnes": 1150.0,
            "normal_seasonal_arrival_tonnes": 1400.0,
            "price_trend_7d_pct": +2.1,
            "primary_transit_corridors": ["NH-44 (Ludhiana-Delhi)", "GT Road"],
        },
        "gujarat": {
            "mandi_name": "Rajkot APMC Cotton & Groundnut Yard",
            "district": "Rajkot",
            "state": "Gujarat",
            "crop": "Cotton (Shankar-6)",
            "modal_price_inr_quintal": 7150.0,
            "min_price_inr_quintal": 6800.0,
            "max_price_inr_quintal": 7500.0,
            "msp_benchmark_inr": 6620.0,
            "daily_arrivals_tonnes": 520.0,
            "normal_seasonal_arrival_tonnes": 700.0,
            "price_trend_7d_pct": +6.8,
            "primary_transit_corridors": ["NH-27 (Saurashtra Highway)", "Kandla Port Corridor"],
        },
    }

    def evaluate_market_impact(
        self,
        region: str = "odisha",
        crop: str = "rice",
        hazard: str = "cyclone",
        efi_intensity: float = 0.92,
    ) -> dict[str, Any]:
        """
        Projects wholesale price fluctuations, arrival deficits, and supply chain disruptions
        resulting from the detected weather hazard.
        """
        reg_key = region.lower().replace(" ", "_")
        mandi = self.MANDI_DATA.get(reg_key, self.MANDI_DATA["odisha"])

        # Weather shock modeling
        # Higher EFI (>0.8) leads to larger supply arrival contraction
        arrival_deficit_pct = float(min(75.0, max(15.0, efi_intensity * 60.0)))
        projected_price_surge_pct = float(min(25.0, max(5.0, efi_intensity * 18.0)))

        volatility_level = "Severe Volatility" if efi_intensity >= 0.85 else "High Volatility" if efi_intensity >= 0.70 else "Moderate Volatility"
        supply_corridor_status = "Inundation & Transport Blockage Expected" if "cyclone" in hazard.lower() or "flood" in hazard.lower() else "Heat Stress Transit Warning"

        return {
            "mandi_id": f"MND-{mandi['state'][:2].upper()}-001",
            "mandi_name": mandi["mandi_name"],
            "location": {
                "district": mandi["district"],
                "state": mandi["state"],
            },
            "commodity": {
                "crop_name": mandi["crop"],
                "modal_price_inr_quintal": mandi["modal_price_inr_quintal"],
                "price_range_inr": [mandi["min_price_inr_quintal"], mandi["max_price_inr_quintal"]],
                "msp_benchmark_inr": mandi["msp_benchmark_inr"],
                "premium_over_msp_pct": round(((mandi["modal_price_inr_quintal"] - mandi["msp_benchmark_inr"]) / mandi["msp_benchmark_inr"]) * 100, 2),
            },
            "arrivals_intelligence": {
                "current_daily_arrivals_tonnes": mandi["daily_arrivals_tonnes"],
                "normal_seasonal_benchmark_tonnes": mandi["normal_seasonal_arrival_tonnes"],
                "arrival_deficit_vs_normal_pct": round(((mandi["normal_seasonal_arrival_tonnes"] - mandi["daily_arrivals_tonnes"]) / mandi["normal_seasonal_arrival_tonnes"]) * 100, 1),
            },
            "weather_shock_forecast": {
                "hazard_driving_shock": hazard.replace("_", " ").title(),
                "efi_severity": round(efi_intensity, 3),
                "projected_arrival_reduction_72h_pct": round(arrival_deficit_pct, 1),
                "projected_price_surge_pct": round(projected_price_surge_pct, 1),
                "volatility_status": volatility_level,
                "supply_corridor_risk": supply_corridor_status,
                "impacted_transit_routes": mandi["primary_transit_corridors"],
            },
            "fpo_and_procurement_advisory": (
                "Expedite pre-landfall procurement; redirect truck freight away from coastal routes; "
                "activate district buffer grain reserves to stabilize mandi retail prices."
            ),
        }


market_engine = MarketIntelligenceEngine()
