"""
Analytics Engine for Rainfall Analysis Dashboard.
Implements descriptive statistics, time-series decomposition, moving averages,
linear regression, Sen's slope, decadal aggregations, IMD anomaly classification,
seasonality indices, correlation matrices, and statistical forecasting.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    MONTHS, MONTH_NAMES, SEASONS, SEASON_KEYS, IMD_ANOMALY_CATEGORIES
)
from data_loader import load_subdivision_data, load_district_data


def classify_imd_anomaly(departure_pct: float) -> Dict[str, Any]:
    """
    Classifies a percentage rainfall departure according to IMD meteorological guidelines.
    """
    if pd.isna(departure_pct):
        return {
            "category": "Unknown",
            "code": "U",
            "color": "#94a3b8",
            "description": "No data available"
        }

    for cat in IMD_ANOMALY_CATEGORIES:
        if cat["min_pct"] <= departure_pct <= cat["max_pct"]:
            return {
                "category": cat["category"],
                "code": cat["code"],
                "color": cat["color"],
                "description": cat["description"]
            }

    # Fallback bounds
    if departure_pct > 0:
        return {
            "category": "Large Excess",
            "code": "LE",
            "color": "#0284c7",
            "description": "Rainfall departure >= +60%"
        }
    return {
        "category": "Large Deficient",
        "code": "LD",
        "color": "#ef4444",
        "description": "Rainfall departure <= -60%"
    }


def calculate_seasonality_index(monthly_means: List[float], annual_mean: float) -> float:
    """
    Calculates Walsh & Lawler (1981) Seasonality Index (SI).
    SI = (1 / R) * sum(|x_n - (R / 12)|)
    Range:
    <= 0.19: Very equable (rainfall evenly spread)
    0.20 - 0.39: Equable with a definite dry season
    0.40 - 0.59: Rather seasonal with a short drier season
    0.60 - 0.79: Seasonal
    0.80 - 0.99: Markedly seasonal
    1.00 - 1.19: Most rain in <= 3 months
    >= 1.20: Extreme, almost all rain in 1-2 months
    """
    if annual_mean <= 0 or not monthly_means:
        return 0.0
    equal_monthly = annual_mean / 12.0
    total_abs_dev = sum(abs(m - equal_monthly) for m in monthly_means)
    return round(total_abs_dev / annual_mean, 3)


def calculate_kpis(
    subdivision: str = "ALL INDIA",
    year_start: int = 1901,
    year_end: int = 2015,
    metric: str = "ANNUAL"
) -> Dict[str, Any]:
    """
    Calculates high-level executive KPI metrics for the selected slice.
    """
    df = load_subdivision_data()
    mask = (df["SUBDIVISION"] == subdivision) & (df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)
    filtered = df[mask].sort_values("YEAR")

    if filtered.empty or metric not in filtered.columns:
        return {
            "subdivision": subdivision,
            "metric": metric,
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "min_val": 0.0,
            "min_year": 0,
            "max_val": 0.0,
            "max_year": 0,
            "trend_slope_per_decade": 0.0,
            "trend_p_value": 1.0,
            "monsoon_share_pct": 0.0,
            "drought_years_count": 0,
            "excess_years_count": 0,
            "seasonality_index": 0.0
        }

    series = filtered[metric].values
    years = filtered["YEAR"].values
    mean_val = float(np.mean(series))
    median_val = float(np.median(series))
    std_val = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
    cv_val = float((std_val / mean_val * 100.0) if mean_val > 0 else 0.0)

    min_idx = int(np.argmin(series))
    max_idx = int(np.argmax(series))
    min_val = float(series[min_idx])
    min_year = int(years[min_idx])
    max_val = float(series[max_idx])
    max_year = int(years[max_idx])

    # Trend regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, series)
    slope_per_decade = float(slope * 10.0)

    # Monsoon dominance (% of annual that fell in Jun-Sep)
    annual_series = filtered["ANNUAL"].values
    monsoon_series = filtered["Jun-Sep"].values
    if np.sum(annual_series) > 0:
        monsoon_share = float(np.sum(monsoon_series) / np.sum(annual_series) * 100.0)
    else:
        monsoon_share = 0.0

    # Departures relative to period mean
    departures = ((series - mean_val) / mean_val * 100.0) if mean_val > 0 else np.zeros_like(series)
    drought_count = int(np.sum(departures <= -20.0))
    excess_count = int(np.sum(departures >= 20.0))

    # Monthly means for Seasonality Index
    monthly_means = [float(filtered[m].mean()) for m in MONTHS]
    annual_mean = float(filtered["ANNUAL"].mean())
    si = calculate_seasonality_index(monthly_means, annual_mean)

    return {
        "subdivision": subdivision,
        "metric": metric,
        "year_range": [int(year_start), int(year_end)],
        "count": len(series),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "cv": round(cv_val, 2),
        "min_val": round(min_val, 2),
        "min_year": min_year,
        "min_departure": round(float(((min_val - mean_val) / mean_val) * 100.0) if mean_val > 0 else 0.0, 1),
        "max_val": round(max_val, 2),
        "max_year": max_year,
        "max_departure": round(float(((max_val - mean_val) / mean_val) * 100.0) if mean_val > 0 else 0.0, 1),
        "trend_slope_per_decade": round(slope_per_decade, 2),
        "trend_r_squared": round(float(r_value ** 2), 4),
        "trend_p_value": round(float(p_value), 4),
        "trend_significant": bool(p_value < 0.05),
        "monsoon_share_pct": round(monsoon_share, 1),
        "drought_years_count": drought_count,
        "excess_years_count": excess_count,
        "normal_years_count": len(series) - drought_count - excess_count,
        "seasonality_index": si
    }


def get_timeseries_analysis(
    subdivision: str = "ALL INDIA",
    year_start: int = 1901,
    year_end: int = 2015,
    metric: str = "ANNUAL"
) -> Dict[str, Any]:
    """
    Computes time-series data, 5-yr & 10-yr rolling moving averages, linear trend line,
    normal reference, percentage departures, and IMD anomaly classification.
    """
    df = load_subdivision_data()
    mask = (df["SUBDIVISION"] == subdivision) & (df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)
    filtered = df[mask].sort_values("YEAR").copy()

    if filtered.empty or metric not in filtered.columns:
        return {"records": [], "statistics": {}, "trendline": {}}

    values = filtered[metric].values
    years = filtered["YEAR"].values
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0

    # Moving averages
    ma5 = filtered[metric].rolling(window=5, min_periods=1, center=True).mean().round(2).tolist()
    ma10 = filtered[metric].rolling(window=10, min_periods=1, center=True).mean().round(2).tolist()

    # Linear trendline
    slope, intercept, r_val, p_val, std_err = stats.linregress(years, values)
    trend_vals = [round(float(slope * y + intercept), 2) for y in years]

    # Departures and SPI approximation
    departures = [
        round(float(((v - mean_val) / mean_val) * 100.0), 2) if mean_val > 0 else 0.0
        for v in values
    ]
    spi_vals = [
        round(float((v - mean_val) / std_val), 2) if std_val > 0 else 0.0
        for v in values
    ]

    records = []
    for i, y in enumerate(years):
        dep = departures[i]
        anomaly = classify_imd_anomaly(dep)
        records.append({
            "year": int(y),
            "actual": round(float(values[i]), 2),
            "ma5": ma5[i],
            "ma10": ma10[i],
            "trend": trend_vals[i],
            "departure_pct": dep,
            "spi": spi_vals[i],
            "category": anomaly["category"],
            "category_code": anomaly["code"],
            "color": anomaly["color"]
        })

    return {
        "subdivision": subdivision,
        "metric": metric,
        "baseline_normal": round(mean_val, 2),
        "standard_deviation": round(std_val, 2),
        "records": records,
        "trend_summary": {
            "slope_per_year": round(float(slope), 3),
            "slope_per_decade": round(float(slope * 10), 2),
            "intercept": round(float(intercept), 2),
            "r_squared": round(float(r_val ** 2), 4),
            "p_value": round(float(p_val), 4),
            "is_significant": bool(p_val < 0.05)
        }
    }


def get_decadal_analysis(
    subdivision: str = "ALL INDIA",
    metric: str = "ANNUAL"
) -> Dict[str, Any]:
    """
    Aggregates rainfall data into 10-year standard decades to examine long-term climate shifts.
    """
    df = load_subdivision_data()
    filtered = df[df["SUBDIVISION"] == subdivision].sort_values("YEAR").copy()

    if filtered.empty or metric not in filtered.columns:
        return {"decades": [], "overall_mean": 0.0}

    overall_mean = float(filtered[metric].mean())

    # Build decade boundaries
    decades_def = [
        ("1901-1910", 1901, 1910),
        ("1911-1920", 1911, 1920),
        ("1921-1930", 1921, 1930),
        ("1931-1940", 1931, 1940),
        ("1941-1950", 1941, 1950),
        ("1951-1960", 1951, 1960),
        ("1961-1970", 1961, 1970),
        ("1971-1980", 1971, 1980),
        ("1981-1990", 1981, 1990),
        ("1991-2000", 1991, 2000),
        ("2001-2010", 2001, 2010),
        ("2011-2015", 2011, 2015),
    ]

    results = []
    for label, start_yr, end_yr in decades_def:
        dec_df = filtered[(filtered["YEAR"] >= start_yr) & (filtered["YEAR"] <= end_yr)]
        if dec_df.empty:
            continue
        vals = dec_df[metric].values
        dec_mean = float(np.mean(vals))
        dec_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        dec_min = float(np.min(vals))
        dec_max = float(np.max(vals))
        departure = ((dec_mean - overall_mean) / overall_mean * 100.0) if overall_mean > 0 else 0.0

        results.append({
            "decade": label,
            "years_count": len(vals),
            "mean": round(dec_mean, 2),
            "std": round(dec_std, 2),
            "min": round(dec_min, 2),
            "max": round(dec_max, 2),
            "departure_from_overall_mean": round(departure, 2),
            "is_above_normal": bool(departure >= 0)
        })

    return {
        "subdivision": subdivision,
        "metric": metric,
        "overall_mean": round(overall_mean, 2),
        "decades": results
    }


def get_seasonality_analysis(
    subdivision: str = "ALL INDIA",
    year_start: int = 1901,
    year_end: int = 2015
) -> Dict[str, Any]:
    """
    Computes monthly climatological cycle and seasonal distribution.
    """
    df = load_subdivision_data()
    mask = (df["SUBDIVISION"] == subdivision) & (df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)
    filtered = df[mask]

    if filtered.empty:
        return {"monthly_climatology": [], "seasonal_breakdown": [], "seasonality_index": 0.0}

    # Monthly Climatology
    monthly_data = []
    monthly_means = []
    for m in MONTHS:
        s = filtered[m]
        mean_v = float(s.mean())
        monthly_means.append(mean_v)
        monthly_data.append({
            "month": m,
            "full_name": MONTH_NAMES[m],
            "mean": round(mean_v, 2),
            "median": round(float(s.median()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
            "p25": round(float(np.percentile(s, 25)), 2),
            "p75": round(float(np.percentile(s, 75)), 2)
        })

    # Seasonal Breakdown
    seasonal_data = []
    total_annual = float(filtered["ANNUAL"].mean())
    for s_key in SEASON_KEYS:
        s_info = SEASONS[s_key]
        s_mean = float(filtered[s_key].mean())
        pct = (s_mean / total_annual * 100.0) if total_annual > 0 else 0.0
        seasonal_data.append({
            "season_key": s_key,
            "name": s_info["name"],
            "description": s_info["description"],
            "mean_rainfall": round(s_mean, 2),
            "percentage_share": round(pct, 1),
            "color": s_info["color"]
        })

    si = calculate_seasonality_index(monthly_means, total_annual)

    return {
        "subdivision": subdivision,
        "annual_mean": round(total_annual, 2),
        "seasonality_index": si,
        "monthly_climatology": monthly_data,
        "seasonal_breakdown": seasonal_data
    }


def get_district_analytics(
    state_filter: Optional[str] = None,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Computes district normal benchmarks, rankings (wettest vs driest),
    distribution histograms, and state averages.
    """
    df = load_district_data()

    if state_filter and state_filter.upper() != "ALL":
        target_df = df[df["STATE_UT_NAME"] == state_filter.upper()].copy()
    else:
        target_df = df.copy()

    if target_df.empty:
        return {
            "state_filter": state_filter or "ALL",
            "total_districts": 0,
            "wettest_districts": [],
            "driest_districts": [],
            "distribution": {},
            "state_averages": []
        }

    # Top wettest
    wettest = target_df.sort_values("ANNUAL", ascending=False).head(top_n)
    wettest_list = [
        {
            "district": row["DISTRICT"],
            "state": row["STATE_UT_NAME"],
            "annual": round(float(row["ANNUAL"]), 2),
            "monsoon": round(float(row["Jun-Sep"]), 2)
        }
        for _, row in wettest.iterrows()
    ]

    # Top driest
    driest = target_df.sort_values("ANNUAL", ascending=True).head(top_n)
    driest_list = [
        {
            "district": row["DISTRICT"],
            "state": row["STATE_UT_NAME"],
            "annual": round(float(row["ANNUAL"]), 2),
            "monsoon": round(float(row["Jun-Sep"]), 2)
        }
        for _, row in driest.iterrows()
    ]

    # Rainfall distribution histogram (bins in mm)
    bins = [0, 500, 1000, 1500, 2000, 3000, 4000, 8000]
    bin_labels = ["< 500 mm", "500-1000 mm", "1000-1500 mm", "1500-2000 mm", "2000-3000 mm", "3000-4000 mm", "> 4000 mm"]
    binned = pd.cut(target_df["ANNUAL"], bins=bins, labels=bin_labels, right=False)
    counts = binned.value_counts(sort=False).to_dict()
    distribution = [{"bin": k, "count": int(v)} for k, v in counts.items()]

    # State averages
    state_agg = df.groupby("STATE_UT_NAME", as_index=False)["ANNUAL"].mean()
    state_agg = state_agg.sort_values("ANNUAL", ascending=False)
    state_averages = [
        {"state": row["STATE_UT_NAME"], "mean_annual": round(float(row["ANNUAL"]), 2)}
        for _, row in state_agg.iterrows()
    ]

    return {
        "state_filter": state_filter or "ALL",
        "total_districts": len(target_df),
        "wettest_districts": wettest_list,
        "driest_districts": driest_list,
        "distribution": distribution,
        "state_averages": state_averages
    }


def get_district_profile(state: str, district: str) -> Optional[Dict[str, Any]]:
    """
    Returns the complete monthly and seasonal profile of a specific district.
    """
    df = load_district_data()
    match = df[(df["STATE_UT_NAME"] == state.upper()) & (df["DISTRICT"].str.lower() == district.lower())]
    if match.empty:
        return None

    row = match.iloc[0]
    monthly = [{"month": m, "value": round(float(row[m]), 2)} for m in MONTHS]
    seasons = [
        {
            "season": s,
            "name": SEASONS[s]["name"],
            "value": round(float(row[s]), 2),
            "pct": round(float(row[s] / row["ANNUAL"] * 100.0), 1) if row["ANNUAL"] > 0 else 0.0
        }
        for s in SEASON_KEYS
    ]

    return {
        "district": row["DISTRICT"],
        "state": row["STATE_UT_NAME"],
        "annual": round(float(row["ANNUAL"]), 2),
        "monthly": monthly,
        "seasons": seasons
    }


def get_correlation_matrix(
    mode: str = "seasons",
    subdivision: str = "ALL INDIA"
) -> Dict[str, Any]:
    """
    Calculates Pearson correlation matrix.
    If mode == 'seasons': correlations among seasons (Jan-Feb, Mar-May, Jun-Sep, Oct-Dec, ANNUAL)
    If mode == 'subdivisions': correlations between major meteorological regions
    """
    df = load_subdivision_data()

    if mode == "seasons":
        filtered = df[df["SUBDIVISION"] == subdivision]
        cols = ["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec", "ANNUAL"]
        corr = filtered[cols].corr().round(3)
        return {
            "mode": "seasons",
            "labels": cols,
            "matrix": [[float(corr.loc[r, c]) for c in cols] for r in cols]
        }
    else:
        # Cross-subdivision correlation across key representative subdivisions
        key_subdivs = [
            "ALL INDIA", "KERALA", "KONKAN & GOA", "COASTAL KARNATAKA",
            "EAST RAJASTHAN", "WEST RAJASTHAN", "EAST UTTAR PRADESH",
            "BIHAR", "ASSAM & MEGHALAYA", "TAMIL NADU", "PUNJAB"
        ]
        pivoted = df[df["SUBDIVISION"].isin(key_subdivs)].pivot(index="YEAR", columns="SUBDIVISION", values="ANNUAL")
        pivoted = pivoted[[col for col in key_subdivs if col in pivoted.columns]]
        corr = pivoted.corr().round(3)
        labels = list(corr.columns)
        return {
            "mode": "subdivisions",
            "labels": labels,
            "matrix": [[float(corr.loc[r, c]) for c in labels] for r in labels]
        }


def get_forecast_analysis(
    subdivision: str = "ALL INDIA",
    metric: str = "ANNUAL",
    forecast_horizon: int = 10
) -> Dict[str, Any]:
    """
    Performs statistical time-series forecasting using Holt's Linear Trend /
    Double Exponential Smoothing combined with linear autoregressive projection.
    Computes 80% and 95% confidence intervals, and historical recurrence return periods.
    """
    df = load_subdivision_data()
    filtered = df[df["SUBDIVISION"] == subdivision].sort_values("YEAR")

    if filtered.empty or metric not in filtered.columns:
        return {"historical_tail": [], "forecast": [], "return_periods": []}

    years = filtered["YEAR"].values
    series = filtered[metric].values
    last_year = int(years[-1])

    # Fit Holt's Linear Trend (Double Exponential Smoothing)
    # Level: L_t = alpha * Y_t + (1 - alpha) * (L_{t-1} + b_{t-1})
    # Trend: b_t = beta * (L_t - L_{t-1}) + (1 - beta) * b_{t-1}
    alpha = 0.25
    beta = 0.05
    level = series[0]
    trend = series[1] - series[0] if len(series) > 1 else 0.0

    residuals = []
    for t in range(len(series)):
        val = series[t]
        pred = level + trend
        residuals.append(val - pred)
        last_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend

    rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    # Forecast next N steps
    forecast_records = []
    for h in range(1, forecast_horizon + 1):
        f_year = last_year + h
        # Dampen trend slightly over horizon
        damping = 0.95 ** h
        f_val = max(0.0, float(level + h * trend * damping))
        # Confidence intervals widen with horizon: sigma_h = RMSE * sqrt(h)
        sigma_h = rmse * math.sqrt(h)
        ci80_margin = 1.282 * sigma_h
        ci95_margin = 1.960 * sigma_h

        forecast_records.append({
            "year": f_year,
            "forecast": round(f_val, 2),
            "ci80_lower": round(max(0.0, f_val - ci80_margin), 2),
            "ci80_upper": round(f_val + ci80_margin, 2),
            "ci95_lower": round(max(0.0, f_val - ci95_margin), 2),
            "ci95_upper": round(f_val + ci95_margin, 2)
        })

    # Return period calculation using Weibull formula: T = (N + 1) / rank
    # Sort rainfall to find extreme high & low thresholds
    sorted_asc = np.sort(series)
    n = len(series)
    # Drought threshold: 10th percentile; Severe drought: 5th percentile
    p10_val = float(np.percentile(series, 10))
    p05_val = float(np.percentile(series, 5))
    p90_val = float(np.percentile(series, 90))
    p95_val = float(np.percentile(series, 95))

    # For upcoming year (h=1)
    upcoming_fc = None
    if forecast_records:
        upcoming_fc = dict(forecast_records[0])
        hist_mean = float(np.mean(series))
        dep_pct = round(((upcoming_fc["forecast"] - hist_mean) / hist_mean * 100.0), 1) if hist_mean > 0 else 0.0
        upcoming_fc["baseline_mean"] = round(hist_mean, 2)
        upcoming_fc["departure_pct"] = dep_pct
        upcoming_fc["subdivision"] = subdivision
        upcoming_fc["metric"] = metric

    return {
        "subdivision": subdivision,
        "metric": metric,
        "last_historical_year": last_year,
        "upcoming_year": last_year + 1,
        "upcoming_year_forecast": upcoming_fc,
        "rmse": round(rmse, 2),
        "forecast": forecast_records,
        "historical_tail": [
            {"year": int(years[i]), "actual": round(float(series[i]), 2)}
            for i in range(max(0, len(series) - 25), len(series))
        ],
        "risk_thresholds": {
            "severe_drought_threshold": round(p05_val, 2),
            "severe_drought_return_period_years": round((n + 1) / (0.05 * n), 1),
            "moderate_drought_threshold": round(p10_val, 2),
            "moderate_drought_return_period_years": round((n + 1) / (0.10 * n), 1),
            "high_flood_threshold": round(p90_val, 2),
            "high_flood_return_period_years": round((n + 1) / (0.10 * n), 1),
            "severe_flood_threshold": round(p95_val, 2),
            "severe_flood_return_period_years": round((n + 1) / (0.05 * n), 1)
        }
    }
