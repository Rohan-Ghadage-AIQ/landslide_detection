"""
early_warning.py
-----------------
Combines the static susceptibility model with dynamic trigger conditions
(forecasted rainfall + antecedent saturation) to produce a ~1-week-ahead
risk score and risk level, per Section 2 of the manager R&D overview.

IMPORTANT: the trigger-scoring formula and reference constants below
(API_SATURATION_REFERENCE_MM, FORECAST_TRIGGER_REFERENCE_MM, RISK_LEVELS)
are reasonable starting defaults, not validated thresholds. They must be
calibrated against real forecast-vs-outcome data collected over at least
one monsoon season before being used operationally.
"""

import numpy as np

import config


def compute_trigger_score(current_api_mm, forecast_rainfall_mm):
    """
    Produces a 0-1 'how close to triggering' score from:
      - current_api_mm       : today's Antecedent Precipitation Index (mm)
      - forecast_rainfall_mm : cumulative rainfall forecast for the next
                                config.FORECAST_LEAD_DAYS days (mm)

    Both terms are min-max scaled against reference constants and averaged.
    This is a simple, transparent starting formula -- swap in a
    learned/calibrated function once real event data is available.
    """
    saturation_term = np.clip(
        current_api_mm / config.API_SATURATION_REFERENCE_MM, 0, 1.5)
    forecast_term = np.clip(
        forecast_rainfall_mm / config.FORECAST_TRIGGER_REFERENCE_MM, 0, 1.5)
    trigger_score = 0.5 * saturation_term + 0.5 * forecast_term
    return float(np.clip(trigger_score, 0, 1.5))


def classify_risk_level(combined_score):
    for low, high, label in config.RISK_LEVELS:
        if low <= combined_score < high:
            return label
    return "Very High"  # anything above the top bound


def compute_combined_risk(susceptibility_prob, current_api_mm,
                           forecast_rainfall_mm):
    """
    susceptibility_prob : model output, 0-1, static "where" score for a
                           given location/pixel.
    Returns (combined_score, risk_level).
    """
    trigger_score = compute_trigger_score(current_api_mm, forecast_rainfall_mm)
    combined_score = float(np.clip(susceptibility_prob * trigger_score, 0, 1))
    return combined_score, classify_risk_level(combined_score)


def compute_combined_risk_raster(susceptibility_raster, current_api_mm,
                                  forecast_rainfall_mm):
    """
    Vectorized version for a full susceptibility raster (2D array), given
    a single district-wide API and forecast value (the simplest version --
    for a real deployment you'd want spatially-varying API/forecast rasters
    rather than one district-wide number).
    """
    trigger_score = compute_trigger_score(current_api_mm, forecast_rainfall_mm)
    combined = np.clip(susceptibility_raster * trigger_score, 0, 1)
    return combined.astype("float32")
