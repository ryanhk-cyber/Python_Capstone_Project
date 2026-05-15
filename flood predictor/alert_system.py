import os
from datetime import datetime

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import OUTPUT_DIR, REPORT_TXT_FILE, RISK_EMOJI
from data.kenya_flood_drainage_dataset import get_worst_subcounties

console = Console()


def summary_panel(risk_df):
    counts = risk_df["risk_level"].value_counts()
    lines = [
        f"{RISK_EMOJI.get(lvl, '')} {lvl}: {counts.get(lvl, 0)} counties"
        for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title=f"Kenya Flood Risk — {datetime.today().strftime('%d %b %Y  %H:%M')}",
            border_style="blue",
        )
    )


def alert_table(risk_df):
    alerted = risk_df[risk_df["risk_level"].isin(["CRITICAL", "HIGH"])].copy()

    if alerted.empty:
        console.print("[green]No HIGH or CRITICAL counties at this time.[/green]")
        return alerted

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
    tbl.add_column("County", min_width=16, style="bold")
    tbl.add_column("Level", min_width=12)
    tbl.add_column("FRI", min_width=6)
    tbl.add_column("7-day rain", min_width=11)
    tbl.add_column("Peak day", min_width=10)
    tbl.add_column("ML prob", min_width=9)
    tbl.add_column("Worst sub-county", min_width=20)
    tbl.add_column("Flood mechanism", min_width=28)

    for _, row in alerted.iterrows():
        level = row["risk_level"]
        color = "red" if level == "CRITICAL" else "yellow"
        worst = get_worst_subcounties(row["county"], top_n=1)
        sub   = worst[0]["Sub_County"] if worst else "—"
        note  = worst[0]["Notes"] if worst else "—"
        prob  = str(row.get("ml_flood_prob", "—"))

        tbl.add_row(
            row["county"],
            f"[{color}]{row['risk_emoji']} {level}[/{color}]",
            str(row["fri_score"]),
            f"{row['forecast_7day_mm']} mm",
            f"{row['peak_daily_mm']} mm",
            prob,
            sub,
            note,
        )

    console.print(tbl)
    return alerted


def save_report(risk_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = [
        "KENYA FLOOD RISK REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 64,
        "",
    ]

    for _, row in risk_df.iterrows():
        worst = get_worst_subcounties(row["county"], top_n=1)
        sub_info = (
            f"{worst[0]['Sub_County']} — {worst[0]['Notes']}" if worst else "—"
        )
        lines += [
            f"County:              {row['county']}",
            f"Risk Level:          {row['risk_level']} (FRI {row['fri_score']})",
            f"7-day Forecast:      {row['forecast_7day_mm']} mm",
            f"Peak Daily:          {row['peak_daily_mm']} mm",
            f"Soil Moisture:       {row['soil_moisture']:.4f} m³/m³",
            f"14-day Antecedent:   {row['antecedent_14day_mm']} mm",
            f"ML Flood Prob:       {row.get('ml_flood_prob', '—')}",
            f"Worst Sub-County:    {sub_info}",
            "-" * 40,
        ]

    with open(REPORT_TXT_FILE, "w") as f:
        f.write("\n".join(lines))
    return REPORT_TXT_FILE


def run_alerts(risk_df):
    summary_panel(risk_df)
    alerted = alert_table(risk_df)
    report = save_report(risk_df)
    console.print(f"\n[dim]Report saved → {report}[/dim]")
    return alerted
