"""
Configuration and Constants for Rainfall Analysis Dashboard Project.
Defines dataset paths, seasonal definitions, meteorological thresholds, and server settings.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DISTRICT_CSV_PATH = BASE_DIR / "district wise rainfall normal.csv"
SUBDIVISION_CSV_PATH = BASE_DIR / "rainfall in india 1901-2015.csv"

# Meteorological Definitions
MONTHS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

MONTH_NAMES = {
    "JAN": "January", "FEB": "February", "MAR": "March",
    "APR": "April", "MAY": "May", "JUN": "June",
    "JUL": "July", "AUG": "August", "SEP": "September",
    "OCT": "October", "NOV": "November", "DEC": "December"
}

SEASONS = {
    "Jan-Feb": {
        "name": "Winter Monsoon",
        "description": "January to February - Post-winter & dry spells",
        "months": ["JAN", "FEB"],
        "color": "#38bdf8"  # Light blue
    },
    "Mar-May": {
        "name": "Pre-Monsoon / Summer",
        "description": "March to May - Thunderstorms and convective rainfall",
        "months": ["MAR", "APR", "MAY"],
        "color": "#f59e0b"  # Amber
    },
    "Jun-Sep": {
        "name": "Southwest Monsoon",
        "description": "June to September - Primary Indian summer monsoon (Kharif)",
        "months": ["JUN", "JUL", "AUG", "SEP"],
        "color": "#10b981"  # Emerald green
    },
    "Oct-Dec": {
        "name": "Post-Monsoon / NE Monsoon",
        "description": "October to December - Northeast monsoon (crucial for SE India)",
        "months": ["OCT", "NOV", "DEC"],
        "color": "#8b5cf6"  # Purple
    }
}

SEASON_KEYS = ["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]

# IMD (India Meteorological Department) Rainfall Anomaly Classification Criteria
# Percentage departure = ((Actual - Normal) / Normal) * 100
IMD_ANOMALY_CATEGORIES = [
    {
        "category": "Large Excess",
        "code": "LE",
        "min_pct": 60.0,
        "max_pct": float("inf"),
        "color": "#0284c7",   # Deep Sky Blue
        "description": "Rainfall departure >= +60%"
    },
    {
        "category": "Excess",
        "code": "E",
        "min_pct": 20.0,
        "max_pct": 59.99,
        "color": "#10b981",   # Emerald Green
        "description": "Rainfall departure between +20% and +59%"
    },
    {
        "category": "Normal",
        "code": "N",
        "min_pct": -19.0,
        "max_pct": 19.99,
        "color": "#64748b",   # Slate Grey
        "description": "Rainfall departure between -19% and +19%"
    },
    {
        "category": "Deficient",
        "code": "D",
        "min_pct": -59.0,
        "max_pct": -19.01,
        "color": "#f59e0b",   # Amber / Orange
        "description": "Rainfall departure between -20% and -59% (Moderate Drought Risk)"
    },
    {
        "category": "Large Deficient",
        "code": "LD",
        "min_pct": -100.0,
        "max_pct": -59.01,
        "color": "#ef4444",   # Crimson Red
        "description": "Rainfall departure <= -60% (Severe Drought Risk)"
    }
]

# Server Settings
SERVER_HOST = "0.0.0.0."
SERVER_PORT = int(os.environ.get("PORT",5000))
DEBUG_MODE = False
