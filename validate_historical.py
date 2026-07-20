"""
validate_historical.py
------------------------
Backtests the early-warning logic against a known historical landslide
event: given the event's location and date, and the ACTUAL daily rainfall
record around that date, reconstruct what the model would have said each
day in the week leading up to the event, and check whether it would have
issued a Watch/Warning with enough lead time.

IMPORTANT CAVEAT: this backtest uses the ACTUAL rainfall that occurred in
the 7 days *after* each test day as a stand-in for "the forecast issued
on that day" -- i.e. it assumes a perfect rainfall forecast. This is the
correct way to test whether the susceptibility model + fusion logic are
useful in principle, but it will over-state real-world performance,
since real IMD 7-day forecasts have their own error. Once real forecast
archives are available, re-run this backtest using the forecast that was
ACTUALLY issued on each day, not the realized rainfall.
"""

import numpy as np
import pandas as pd

from feature_engineering import compute_api_series
from early_warning import compute_combined_risk
import config


def extract_feature_vector_at_point(x, y, feature_stack, transform,
                                     feature_cols):
    """Pulls a single feature vector at (x, y) from the raster stack."""
    col, row = ~transform * (x, y)
    row, col = int(row), int(col)
    any_arr = next(iter(feature_stack.values()))
    if not (0 <= row < any_arr.shape[0] and 0 <= col < any_arr.shape[1]):
        raise ValueError("Event location falls outside the raster extent.")
    return np.array([[feature_stack[c][row, col] for c in feature_cols]])


def backtest_event(event_x, event_y, event_date, daily_rainfall: pd.Series,
                    model, feature_stack, transform, feature_cols,
                    lookback_days=10, lead_days=None):
    """
    event_x, event_y : projected coordinates of the historical event,
                        same CRS as the feature rasters.
    event_date        : pandas Timestamp / date-like.
    daily_rainfall     : pandas Series of daily rainfall (mm), indexed by
                          date, covering at least
                          [event_date - lookback_days - API_WINDOW,
                           event_date + lead_days].
    model               : trained classifier with .predict_proba().
    feature_stack, transform, feature_cols : as elsewhere in the pipeline.

    Returns a DataFrame with one row per day in the lookback window:
    date, api_mm, forecast_next_7d_mm, susceptibility_prob,
    combined_score, risk_level, days_before_event.
    """
    lead_days = lead_days or config.FORECAST_LEAD_DAYS
    event_date = pd.Timestamp(event_date)

    api_series = compute_api_series(
        daily_rainfall, k=config.API_DECAY_K, window=config.API_WINDOW_DAYS)

    x_vec = extract_feature_vector_at_point(
        event_x, event_y, feature_stack, transform, feature_cols)
    susceptibility_prob = float(model.predict_proba(x_vec)[0, 1])

    rows = []
    for offset in range(lookback_days, -1, -1):
        test_date = event_date - pd.Timedelta(days=offset)
        if test_date not in api_series.index:
            continue
        current_api = api_series.loc[test_date]
        if np.isnan(current_api):
            continue

        forecast_window = daily_rainfall.loc[
            test_date + pd.Timedelta(days=1):
            test_date + pd.Timedelta(days=lead_days)
        ]
        forecast_rainfall_mm = float(forecast_window.sum())

        combined_score, risk_level = compute_combined_risk(
            susceptibility_prob, current_api, forecast_rainfall_mm)

        rows.append({
            "date": test_date,
            "days_before_event": offset,
            "api_mm": round(float(current_api), 1),
            "forecast_next_7d_mm": round(forecast_rainfall_mm, 1),
            "susceptibility_prob": round(susceptibility_prob, 3),
            "combined_score": round(combined_score, 3),
            "risk_level": risk_level,
        })

    return pd.DataFrame(rows)


def summarize_backtest(trace_df, warning_levels=("High", "Very High")):
    """
    Reports whether the model would have issued a Watch/Warning ahead of
    the event, and with how much lead time.
    """
    if trace_df.empty:
        return {
            "would_have_warned": False,
            "max_lead_time_days": 0,
            "message": "No historical weather data available for the chosen date range.",
        }
        
    escalated = trace_df[trace_df["risk_level"].isin(warning_levels)]
    if escalated.empty:
        return {
            "would_have_warned": False,
            "max_lead_time_days": 0,
            "message": "No High/Very High risk level was reached in the "
                       "lookback window -- the model would NOT have "
                       "flagged this event under current thresholds.",
        }
    max_lead = int(escalated["days_before_event"].max())
    return {
        "would_have_warned": True,
        "max_lead_time_days": max_lead,
        "message": f"Model would have escalated to High/Very High risk "
                   f"{max_lead} day(s) before the event.",
    }
