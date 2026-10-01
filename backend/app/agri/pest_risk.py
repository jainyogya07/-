"""
Module 14: Pest & Disease Risk Engine.
Combines atmospheric variables (temperature, relative humidity, precipitation, wind)
with crop type and growth stage to estimate infection probabilities and environmental drivers.
"""

from typing import Any
import numpy as np


class PestDiseaseRiskEngine:
    """
    Evaluates microclimatic suitability and physical transmission vectors
    for major crop pathogens and pests across India.
    """

    CROP_PROFILES = {
        "rice": {
            "name": "Rice (Paddy)",
            "stages": ["sowing_nursery", "vegetative", "flowering", "maturity"],
            "pathogens": [
                {
                    "name": "Bacterial Leaf Blight (Xanthomonas oryzae)",
                    "type": "bacterial",
                    "ideal_temp": (25.0, 34.0),
                    "min_rh": 80.0,
                    "wind_rain_driven": True,
                    "susceptible_stages": ["vegetative", "flowering"],
                    "symptoms": "Water-soaked lesions turning yellow-white along leaf margins with bacterial ooze.",
                    "advisory": "Avoid nitrogen top-dressing; apply Streptocycline (0.01%) + Copper Oxychloride (0.25%).",
                },
                {
                    "name": "Brown Plant Hopper (Nilaparvata lugens)",
                    "type": "insect_pest",
                    "ideal_temp": (24.0, 32.0),
                    "min_rh": 75.0,
                    "wind_rain_driven": False,
                    "susceptible_stages": ["vegetative", "flowering", "maturity"],
                    "symptoms": "Hopper burn, drying of tillers in circular patches, transmission of grassy stunt virus.",
                    "advisory": "Drain water for 3-4 days; spray Triflumezopyrim 10% SC (94 ml/acre) at base of plants.",
                },
                {
                    "name": "Sheath Blight (Rhizoctonia solani)",
                    "type": "fungal",
                    "ideal_temp": (28.0, 32.0),
                    "min_rh": 85.0,
                    "wind_rain_driven": False,
                    "susceptible_stages": ["vegetative", "flowering"],
                    "symptoms": "Greenish-grey water-soaked spots on leaf sheaths forming snake-skin patterns.",
                    "advisory": "Foliar spray with Hexaconazole 5% EC (2 ml/litre) or Azoxystrobin.",
                },
            ],
        },
        "cotton": {
            "name": "Cotton",
            "stages": ["seedling", "vegetative", "square_flowering", "boll_development"],
            "pathogens": [
                {
                    "name": "Pink Bollworm (Pectinophora gossypiella)",
                    "type": "insect_pest",
                    "ideal_temp": (22.0, 35.0),
                    "min_rh": 65.0,
                    "wind_rain_driven": False,
                    "susceptible_stages": ["square_flowering", "boll_development"],
                    "symptoms": "Rosetted flowers, bore holes in developing bolls plugged with excreta.",
                    "advisory": "Install pheromone traps (5/acre); spray Profenofos 50% EC or Chlorantraniliprole.",
                },
                {
                    "name": "Bacterial Blight / Black Arm (Xanthomonas)",
                    "type": "bacterial",
                    "ideal_temp": (26.0, 32.0),
                    "min_rh": 85.0,
                    "wind_rain_driven": True,
                    "susceptible_stages": ["seedling", "vegetative", "boll_development"],
                    "symptoms": "Angular leaf spots with water-soaked margins extending into black arm stem lesions.",
                    "advisory": "Spray Copper Oxychloride 50 WP (2.5 g/L) mixed with Streptomycin (100 mg/L).",
                },
            ],
        },
        "wheat": {
            "name": "Wheat",
            "stages": ["crown_root", "tillering", "jointing_booting", "heading_grain"],
            "pathogens": [
                {
                    "name": "Yellow / Stripe Rust (Puccinia striiformis)",
                    "type": "fungal",
                    "ideal_temp": (10.0, 20.0),
                    "min_rh": 80.0,
                    "wind_rain_driven": True,
                    "susceptible_stages": ["tillering", "jointing_booting", "heading_grain"],
                    "symptoms": "Yellow pustules arranged in linear stripes on leaf blades.",
                    "advisory": "Foliar spray with Tebuconazole 25.9% EC (1 ml/L) or Propiconazole 25% EC.",
                },
            ],
        },
    }

    def evaluate_risk(
        self,
        crop: str = "rice",
        stage: str = "vegetative",
        temp_c: float = 29.5,
        rh_pct: float = 88.0,
        rain_mm: float = 65.0,
        wind_kmh: float = 45.0,
        region: str = "Odisha",
    ) -> dict[str, Any]:
        """
        Calculates multi-pathogen infection probabilities based on atmospheric drivers.
        """
        crop_key = crop.lower().strip()
        profile = self.CROP_PROFILES.get(crop_key, self.CROP_PROFILES["rice"])

        pathogen_results = []
        risk_scores = []

        for p in profile["pathogens"]:
            score = 0.30  # Baseline ambient pressure

            # 1. Temperature suitability
            t_min, t_max = p["ideal_temp"]
            if t_min <= temp_c <= t_max:
                score += 0.25
            elif abs(temp_c - t_min) <= 4.0 or abs(temp_c - t_max) <= 4.0:
                score += 0.10

            # 2. Relative humidity suitability
            if rh_pct >= p["min_rh"]:
                score += 0.25
            elif rh_pct >= (p["min_rh"] - 15.0):
                score += 0.12

            # 3. Wind & rainfall physical transmission vectors
            if p["wind_rain_driven"] and (rain_mm > 25.0 or wind_kmh > 30.0):
                score += 0.15

            # 4. Growth stage vulnerability
            if stage.lower() in p["susceptible_stages"]:
                score += 0.05

            final_score = float(np.clip(score, 0.10, 0.96))
            risk_scores.append(final_score)

            severity = "critical" if final_score >= 0.80 else "high" if final_score >= 0.65 else "moderate" if final_score >= 0.40 else "low"

            pathogen_results.append({
                "pathogen": p["name"],
                "type": p["type"],
                "risk_probability": round(final_score, 2),
                "severity": severity,
                "contributing_factors": {
                    "temperature_match": bool(t_min <= temp_c <= t_max),
                    "humidity_exceeded": bool(rh_pct >= p["min_rh"]),
                    "wind_rain_vector_active": bool(p["wind_rain_driven"] and (rain_mm > 25.0 or wind_kmh > 30.0)),
                    "stage_susceptible": bool(stage.lower() in p["susceptible_stages"]),
                },
                "symptoms": p["symptoms"],
                "advisory": p["advisory"],
            })

        max_risk = float(max(risk_scores)) if risk_scores else 0.50
        overall_severity = "critical" if max_risk >= 0.80 else "high" if max_risk >= 0.65 else "moderate"

        return {
            "crop": profile["name"],
            "region": region.title(),
            "growth_stage": stage.replace("_", " ").title(),
            "environmental_conditions": {
                "temperature_celsius": round(temp_c, 1),
                "relative_humidity_pct": round(rh_pct, 1),
                "rainfall_24h_mm": round(rain_mm, 1),
                "wind_speed_kmh": round(wind_kmh, 1),
            },
            "overall_pest_disease_risk": round(max_risk, 2),
            "threat_level": overall_severity,
            "action_urgency": "Immediate preventive spray within 24-48 hours" if max_risk >= 0.70 else "Routine surveillance",
            "pathogens_evaluated": pathogen_results,
        }


pest_engine = PestDiseaseRiskEngine()
