"""
Automated Test Suite for Rainfall Analysis Project.
Verifies data loading, statistical calculations, forecast confidence intervals,
and Flask REST API endpoints.
"""

import unittest
import json
from pathlib import Path
import pandas as pd

from config import DISTRICT_CSV_PATH, SUBDIVISION_CSV_PATH, MONTHS, SEASONS
import data_loader
import analytics
from app import app


class TestRainfallAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Record file sizes to ensure original CSVs are never mutated
        cls.orig_dist_size = DISTRICT_CSV_PATH.stat().st_size
        cls.orig_sub_size = SUBDIVISION_CSV_PATH.stat().st_size

    def test_original_files_preserved(self):
        """Ensures source CSV files exist and their sizes have not been altered."""
        self.assertTrue(DISTRICT_CSV_PATH.exists(), "district wise rainfall normal.csv must exist")
        self.assertTrue(SUBDIVISION_CSV_PATH.exists(), "rainfall in india 1901-2015.csv must exist")
        self.assertEqual(DISTRICT_CSV_PATH.stat().st_size, self.orig_dist_size, "District CSV must be unmodified")
        self.assertEqual(SUBDIVISION_CSV_PATH.stat().st_size, self.orig_sub_size, "Subdivision CSV must be unmodified")

    def test_data_loader_metadata(self):
        """Verifies metadata extraction for UI controls."""
        meta = data_loader.get_metadata()
        self.assertIn("subdivisions", meta)
        self.assertIn("states", meta)
        self.assertIn("ALL INDIA", meta["subdivisions"])
        self.assertEqual(meta["total_districts"], 641)
        self.assertEqual(meta["min_year"], 1901)
        self.assertEqual(meta["max_year"], 2015)

    def test_data_loader_subdivision(self):
        """Verifies subdivision dataset cleaning and ALL INDIA synthesis."""
        df = data_loader.load_subdivision_data()
        self.assertFalse(df.empty)
        self.assertIn("ALL INDIA", df["SUBDIVISION"].values)
        # Check non-null values
        self.assertFalse(df["ANNUAL"].isnull().any(), "Imputation should resolve missing values")
        self.assertFalse(df["Jun-Sep"].isnull().any())

    def test_data_loader_district(self):
        """Verifies district dataset shape and columns."""
        df = data_loader.load_district_data()
        self.assertEqual(len(df), 641)
        self.assertEqual(len(df["STATE_UT_NAME"].unique()), 35)

    def test_kpi_calculation(self):
        """Verifies high-level KPI computations."""
        kpi = analytics.calculate_kpis("ALL INDIA", 1901, 2015, "ANNUAL")
        self.assertGreater(kpi["mean"], 1000.0)
        self.assertGreater(kpi["std"], 0.0)
        self.assertEqual(kpi["min_year"], 1972)
        self.assertEqual(kpi["max_year"], 1961)
        self.assertGreater(kpi["monsoon_share_pct"], 60.0)
        self.assertIn("seasonality_index", kpi)

    def test_timeseries_analysis(self):
        """Verifies time-series moving averages and regression."""
        ts = analytics.get_timeseries_analysis("ALL INDIA", 1901, 2015, "ANNUAL")
        records = ts["records"]
        self.assertEqual(len(records), 115)
        for r in records[:5]:
            self.assertIn("year", r)
            self.assertIn("actual", r)
            self.assertIn("ma5", r)
            self.assertIn("trend", r)
            self.assertIn("departure_pct", r)
            self.assertIn("category", r)

    def test_decadal_analysis(self):
        """Verifies 10-year decadal grouping."""
        dec = analytics.get_decadal_analysis("ALL INDIA", "ANNUAL")
        self.assertEqual(len(dec["decades"]), 12)
        self.assertEqual(dec["decades"][0]["decade"], "1901-1910")
        self.assertEqual(dec["decades"][-1]["decade"], "2011-2015")

    def test_seasonality_analysis(self):
        """Verifies 12-month climatology cycle and seasonal shares."""
        sea = analytics.get_seasonality_analysis("ALL INDIA", 1901, 2015)
        self.assertEqual(len(sea["monthly_climatology"]), 12)
        self.assertEqual(len(sea["seasonal_breakdown"]), 4)
        total_share = sum(s["percentage_share"] for s in sea["seasonal_breakdown"])
        self.assertAlmostEqual(total_share, 100.0, delta=1.0)

    def test_forecasting_bounds(self):
        """Verifies statistical forecast confidence intervals widen properly."""
        fc = analytics.get_forecast_analysis("ALL INDIA", "ANNUAL", forecast_horizon=5)
        forecast_list = fc["forecast"]
        self.assertEqual(len(forecast_list), 5)
        for item in forecast_list:
            self.assertLessEqual(item["ci95_lower"], item["ci80_lower"])
            self.assertLessEqual(item["ci80_lower"], item["forecast"])
            self.assertGreaterEqual(item["ci95_upper"], item["ci80_upper"])
            self.assertGreaterEqual(item["ci80_upper"], item["forecast"])

    def test_flask_api_endpoints(self):
        """Verifies Flask HTTP endpoints respond with 200 and valid JSON."""
        client = app.test_client()

        # HTML Home
        res = client.get("/")
        self.assertEqual(res.status_code, 200)

        # Health
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)

        # Metadata
        res = client.get("/api/metadata")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("subdivisions", data)

        # KPI
        res = client.get("/api/kpi?subdivision=KERALA&metric=ANNUAL")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["subdivision"], "KERALA")

        # Districts
        res = client.get("/api/districts?state=MAHARASHTRA")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertGreater(len(data["wettest_districts"]), 0)

        # Table
        res = client.get("/api/table?dataset=subdivision&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data["records"]), 10)

        # Export CSV
        res = client.get("/api/export?dataset=subdivision&subdivision=ALL INDIA")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")


if __name__ == "__main__":
    unittest.main()
