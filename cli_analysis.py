"""
Command-Line Interface (CLI) for Rainfall Analysis Project.
Provides rapid terminal-based climate reports, statistical summaries,
and data exports without needing a web browser.
"""

import argparse
import json
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import MONTHS, SEASON_KEYS
import data_loader
import analytics


def print_banner():
    print("=" * 72)
    print("   [+] HYDROCLIM INDIA: RAINFALL & MONSOON CLIMATE INTELLIGENCE")
    print("       Analysis of 1901-2015 Subdivisions & 641 District Normals")
    print("=" * 72)


def run_cli():
    parser = argparse.ArgumentParser(
        description="Rainfall Analysis Command-Line Intelligence Tool"
    )
    parser.add_argument("--subdivision", type=str, default="ALL INDIA", help="Subdivision name (e.g. 'KERALA', 'ALL INDIA')")
    parser.add_argument("--metric", type=str, default="ANNUAL", help="Rainfall metric ('ANNUAL', 'Jun-Sep', 'JAN', etc.)")
    parser.add_argument("--start", type=int, default=1901, help="Start year (1901-2015)")
    parser.add_argument("--end", type=int, default=2015, help="End year (1901-2015)")
    parser.add_argument("--state", type=str, default=None, help="State name for district analytics")
    parser.add_argument("--forecast", type=int, default=5, help="Forecast horizon years (default: 5)")
    parser.add_argument("--export-json", type=str, default=None, help="Export summary report to a JSON file")

    args = parser.parse_args()
    print_banner()

    print(f"\n[+] Computing KPIs for: {args.subdivision} ({args.start}-{args.end}) | Metric: {args.metric}")
    kpi = analytics.calculate_kpis(args.subdivision, args.start, args.end, args.metric)

    print("-" * 72)
    print(f"  Climatological Mean:     {kpi['mean']:>10.2f} mm")
    print(f"  Standard Deviation:      {kpi['std']:>10.2f} mm")
    print(f"  Coeff. of Variation (CV):{kpi['cv']:>10.2f} %")
    print(f"  Wettest Record ({kpi['max_year']}):   {kpi['max_val']:>10.2f} mm ({kpi['max_departure']:>+.1f}% departure)")
    print(f"  Driest Record ({kpi['min_year']}):    {kpi['min_val']:>10.2f} mm ({kpi['min_departure']:>+.1f}% departure)")
    print(f"  Trend Slope:             {kpi['trend_slope_per_decade']:>10.2f} mm/decade (p={kpi['trend_p_value']}, R2={kpi['trend_r_squared']})")
    print(f"  SW Monsoon Share:        {kpi['monsoon_share_pct']:>10.1f} %")
    print(f"  Seasonality Index (SI):  {kpi['seasonality_index']:>10.3f}")
    print(f"  IMD Deficit / Droughts:  {kpi['drought_years_count']:>10d} years")
    print(f"  IMD Excess Years:        {kpi['excess_years_count']:>10d} years")
    print("-" * 72)

    # Seasonality
    print(f"\n[+] Seasonal Breakdown ({args.subdivision}):")
    season = analytics.get_seasonality_analysis(args.subdivision, args.start, args.end)
    for s in season["seasonal_breakdown"]:
        print(f"  - {s['name']:<28}: {s['mean_rainfall']:>8.2f} mm ({s['percentage_share']:>5.1f}%)")

    # Districts
    print(f"\n[+] District Normal Extremes (State Filter: {args.state or 'ALL INDIA'}):")
    dist = analytics.get_district_analytics(args.state, top_n=5)
    print("  Top 5 Wettest Districts:")
    for d in dist["wettest_districts"]:
        print(f"    * {d['district']} ({d['state']}): {d['annual']} mm (Monsoon: {d['monsoon']} mm)")
    print("  Top 5 Driest Districts:")
    for d in dist["driest_districts"]:
        print(f"    * {d['district']} ({d['state']}): {d['annual']} mm (Monsoon: {d['monsoon']} mm)")

    # Forecast
    if args.forecast > 0:
        print(f"\n[+] Statistical Forecast ({args.forecast} Horizon Years):")
        fc = analytics.get_forecast_analysis(args.subdivision, args.metric, args.forecast)
        for f in fc["forecast"]:
            print(f"  Year {f['year']}: Forecast = {f['forecast']:>8.2f} mm | 95% CI: [{f['ci95_lower']:>8.2f}, {f['ci95_upper']:>8.2f}]")

    # Export JSON if requested
    if args.export_json:
        report_data = {
            "kpi": kpi,
            "seasonality": season,
            "districts": dist,
            "forecast": fc if args.forecast > 0 else None
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[+] Exported summary report successfully to: {args.export_json}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    run_cli()
