import os
from datetime import datetime, timedelta

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    FORECAST_CHART_FILE,
    MAP_HTML_FILE,
    OUTPUT_DIR,
    RAINFALL_THRESHOLDS,
    RISK_CHART_FILE,
)


def create_risk_map(risk_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    m = folium.Map(location=[-0.023, 37.906], zoom_start=6, tiles="CartoDB positron")

    radius_by_level = {"LOW": 7, "MEDIUM": 9, "HIGH": 12, "CRITICAL": 16}

    for _, row in risk_df.iterrows():
        popup_html = (
            f"<div style='font-family:sans-serif;min-width:160px'>"
            f"<b style='font-size:14px'>{row['county']}</b><br>"
            f"<span style='color:{row['risk_color']};font-weight:bold'>"
            f"{row['risk_emoji']} {row['risk_level']}</span><br>"
            f"<hr style='margin:4px 0'>"
            f"FRI Score: <b>{row['fri_score']}</b> / 100<br>"
            f"7-day rain: <b>{row['forecast_7day_mm']} mm</b><br>"
            f"Peak day: <b>{row['peak_daily_mm']} mm</b><br>"
            f"Soil moisture: <b>{row['soil_moisture']:.3f} m³/m³</b><br>"
            f"14-day antecedent: <b>{row['antecedent_14day_mm']} mm</b><br>"
            f"ML flood probability: <b>{row.get('ml_flood_prob', '—')}</b>"
            f"</div>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius_by_level.get(row["risk_level"], 9),
            color=row["risk_color"],
            fill=True,
            fill_color=row["risk_color"],
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{row['county']} — {row['risk_level']} (FRI: {row['fri_score']})",
        ).add_to(m)

    legend = (
        "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
        "background:white;padding:12px 16px;border-radius:8px;"
        "box-shadow:0 2px 8px rgba(0,0,0,0.15);font-family:sans-serif;font-size:13px'>"
        "<b>Flood Risk Level</b><br>"
        "<span style='color:#7b241c'>●</span> Critical &nbsp;"
        "<span style='color:#e74c3c'>●</span> High &nbsp;"
        "<span style='color:#f39c12'>●</span> Medium &nbsp;"
        "<span style='color:#2ecc71'>●</span> Low"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend))
    m.save(MAP_HTML_FILE)
    return MAP_HTML_FILE


def create_risk_chart(risk_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    top = risk_df.head(20).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top["county"], top["fri_score"], color=top["risk_color"], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Flood Risk Index Score (0–100)", fontsize=11)
    ax.set_title(
        f"Top 20 Counties by Flood Risk — {datetime.today().strftime('%d %b %Y')}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    for threshold, color in [(35, "#2ecc71"), (55, "#f39c12"), (75, "#e74c3c")]:
        ax.axvline(threshold, color=color, linestyle="--", alpha=0.45, linewidth=1)

    for bar, score in zip(bars, top["fri_score"]):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            str(score),
            va="center", ha="left", fontsize=9,
        )

    patches = [
        mpatches.Patch(color="#2ecc71", label="Low (0–35)"),
        mpatches.Patch(color="#f39c12", label="Medium (35–55)"),
        mpatches.Patch(color="#e74c3c", label="High (55–75)"),
        mpatches.Patch(color="#7b241c", label="Critical (75–100)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.savefig(RISK_CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    return RISK_CHART_FILE


def create_forecast_chart(risk_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alert_counties = risk_df[risk_df["risk_level"].isin(["CRITICAL", "HIGH"])].head(8)
    if alert_counties.empty:
        # If no HIGH/CRITICAL counties exist, still produce a useful chart
        # by plotting the top 8 counties by FRI score.
        alert_counties = risk_df.head(8)

    today = datetime.today().date()
    day_labels = [(today + timedelta(days=i)).strftime("%a %d") for i in range(7)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in alert_counties.iterrows():
        forecast = row.get("daily_forecast", [])
        if isinstance(forecast, list) and len(forecast) == 7:
            ax.plot(day_labels, forecast, marker="o", linewidth=2,
                    label=row["county"], color=row["risk_color"])

    ax.axhline(
        RAINFALL_THRESHOLDS["peak_heavy"], color="#f39c12",
        linestyle="--", alpha=0.6, linewidth=1, label="Heavy (40 mm)",
    )
    ax.axhline(
        RAINFALL_THRESHOLDS["peak_extreme"], color="#e74c3c",
        linestyle="--", alpha=0.6, linewidth=1, label="Extreme (60 mm)",
    )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Rainfall (mm)", fontsize=11)
    ax.set_title("7-Day Rainfall Forecast — High & Critical Risk Counties", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(FORECAST_CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    return FORECAST_CHART_FILE
