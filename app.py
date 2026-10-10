"""
Flask Application Server for Rainfall Analysis Dashboard Project.
Serves interactive dashboard frontend and comprehensive RESTful analytics APIs.
"""

import io
import csv
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, Response, send_file
import pandas as pd

from config import (
    SERVER_HOST, SERVER_PORT, DEBUG_MODE,
    MONTHS, MONTH_NAMES, SEASONS, SEASON_KEYS, IMD_ANOMALY_CATEGORIES
)
import data_loader
import analytics

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    """Main dashboard page."""
    meta = data_loader.get_metadata()
    return render_template("index.html", subdivisions=meta["subdivisions"])


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Rainfall Analysis Dashboard API",
        "version": "2.0.0"
    })


@app.route("/api/metadata", methods=["GET"])
def metadata():
    """Returns dropdown metadata, state-to-district trees, and classification rules."""
    meta = data_loader.get_metadata()
    return jsonify({
        **meta,
        "month_names": MONTH_NAMES,
        "seasons_info": SEASONS,
        "imd_categories": IMD_ANOMALY_CATEGORIES
    })


@app.route("/api/kpi", methods=["GET"])
def kpi():
    """Returns high-level KPI cards for selected filters."""
    subdivision = request.args.get("subdivision", "ALL INDIA")
    year_start = int(request.args.get("year_start", 1901))
    year_end = int(request.args.get("year_end", 2015))
    metric = request.args.get("metric", "ANNUAL")

    result = analytics.calculate_kpis(subdivision, year_start, year_end, metric)
    return jsonify(result)


@app.route("/api/timeseries", methods=["GET"])
def timeseries():
    """Returns time-series data, moving averages, trendline, departures, and SPI."""
    subdivision = request.args.get("subdivision", "ALL INDIA")
    year_start = int(request.args.get("year_start", 1901))
    year_end = int(request.args.get("year_end", 2015))
    metric = request.args.get("metric", "ANNUAL")

    result = analytics.get_timeseries_analysis(subdivision, year_start, year_end, metric)
    return jsonify(result)


@app.route("/api/decadal", methods=["GET"])
def decadal():
    """Returns 10-year decadal aggregation analysis."""
    subdivision = request.args.get("subdivision", "ALL INDIA")
    metric = request.args.get("metric", "ANNUAL")

    result = analytics.get_decadal_analysis(subdivision, metric)
    return jsonify(result)


@app.route("/api/seasonality", methods=["GET"])
def seasonality():
    """Returns 12-month climatology and seasonal contribution shares."""
    subdivision = request.args.get("subdivision", "ALL INDIA")
    year_start = int(request.args.get("year_start", 1901))
    year_end = int(request.args.get("year_end", 2015))

    result = analytics.get_seasonality_analysis(subdivision, year_start, year_end)
    return jsonify(result)


@app.route("/api/districts", methods=["GET"])
def districts():
    """Returns district normal benchmarks, wettest/driest rankings, and histogram."""
    state = request.args.get("state", "ALL")
    top_n = int(request.args.get("top_n", 10))

    result = analytics.get_district_analytics(state_filter=state, top_n=top_n)
    return jsonify(result)


@app.route("/api/district-profile", methods=["GET"])
def district_profile():
    """Returns detailed monthly profile for a specific district."""
    state = request.args.get("state", "")
    district = request.args.get("district", "")

    result = analytics.get_district_profile(state, district)
    if result is None:
        return jsonify({"error": "District not found"}), 404
    return jsonify(result)


@app.route("/api/correlation", methods=["GET"])
def correlation():
    """Returns correlation matrix for seasons or subdivisions."""
    mode = request.args.get("mode", "seasons")
    subdivision = request.args.get("subdivision", "ALL INDIA")

    result = analytics.get_correlation_matrix(mode=mode, subdivision=subdivision)
    return jsonify(result)


@app.route("/api/forecast", methods=["GET"])
def forecast():
    """Returns statistical forecast and extreme event return periods."""
    subdivision = request.args.get("subdivision", "ALL INDIA")
    metric = request.args.get("metric", "ANNUAL")
    horizon = int(request.args.get("horizon", 10))

    result = analytics.get_forecast_analysis(subdivision, metric, forecast_horizon=horizon)
    return jsonify(result)


@app.route("/api/table", methods=["GET"])
def table():
    """Returns paginated, searchable, and sortable dataset rows."""
    dataset = request.args.get("dataset", "subdivision")
    page = max(1, int(request.args.get("page", 1)))
    page_size = max(5, min(100, int(request.args.get("page_size", 15))))
    search = request.args.get("search", "").strip().lower()
    sort_by = request.args.get("sort_by", "")
    sort_dir = request.args.get("sort_dir", "asc").lower()

    if dataset == "district":
        df = data_loader.load_district_data()
        state = request.args.get("state", "ALL")
        if state and state.upper() != "ALL":
            df = df[df["STATE_UT_NAME"] == state.upper()]

        if search:
            mask = (
                df["DISTRICT"].str.lower().str.contains(search, na=False) |
                df["STATE_UT_NAME"].str.lower().str.contains(search, na=False)
            )
            df = df[mask]
    else:
        df = data_loader.load_subdivision_data()
        subdivision = request.args.get("subdivision", "ALL")
        if subdivision and subdivision.upper() != "ALL":
            df = df[df["SUBDIVISION"] == subdivision]
        year_start = int(request.args.get("year_start", 1901))
        year_end = int(request.args.get("year_end", 2015))
        df = df[(df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)]

        if search:
            mask = (
                df["SUBDIVISION"].str.lower().str.contains(search, na=False) |
                df["YEAR"].astype(str).str.contains(search, na=False)
            )
            df = df[mask]

    total_records = len(df)

    if sort_by and sort_by in df.columns:
        ascending = (sort_dir == "asc")
        df = df.sort_values(by=sort_by, ascending=ascending)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    sliced = df.iloc[start_idx:end_idx]

    return jsonify({
        "dataset": dataset,
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": max(1, (total_records + page_size - 1) // page_size),
        "columns": list(df.columns),
        "records": sliced.to_dict(orient="records")
    })


@app.route("/api/export", methods=["GET"])
def export_csv():
    """Generates and streams filtered CSV download without touching original files."""
    dataset = request.args.get("dataset", "subdivision")

    if dataset == "district":
        df = data_loader.load_district_data()
        state = request.args.get("state", "ALL")
        if state and state.upper() != "ALL":
            df = df[df["STATE_UT_NAME"] == state.upper()]
        filename = f"district_rainfall_filtered_{state}.csv"
    else:
        df = data_loader.load_subdivision_data()
        subdivision = request.args.get("subdivision", "ALL")
        if subdivision and subdivision.upper() != "ALL":
            df = df[df["SUBDIVISION"] == subdivision]
        year_start = int(request.args.get("year_start", 1901))
        year_end = int(request.args.get("year_end", 2015))
        df = df[(df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)]
        clean_name = subdivision.replace(" ", "_").replace("&", "and")
        filename = f"rainfall_subdivision_{clean_name}_{year_start}_{year_end}.csv"

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


if __name__ == "__main__":
    print(f"Starting Rainfall Analysis Dashboard on http://{SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE, threaded=True)
