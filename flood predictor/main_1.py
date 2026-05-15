import os

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def setup():
    for d in ["outputs", "data"]:
        os.makedirs(d, exist_ok=True)


def main():
    setup()
    console.print("\n[bold blue]Kenya Flood Risk Prediction System[/bold blue]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:

        t1 = progress.add_task("Fetching live rainfall data for 47 counties...", total=None)
        from data_loader import load_all_counties
        rainfall_df = load_all_counties(use_cache=True)
        progress.update(t1, description="[green]Rainfall data loaded[/green]")

        t2 = progress.add_task("Calculating Flood Risk Index scores...", total=None)
        from risk_calculator import calculate_all_risks
        risk_df = calculate_all_risks(rainfall_df)
        progress.update(t2, description="[green]FRI scores calculated[/green]")

        t3 = progress.add_task("Running Random Forest predictions...", total=None)
        from flood_predictor import predict_flood_probability
        predictions = predict_flood_probability(risk_df)
        risk_df = risk_df.merge(predictions, on="county", how="left")
        progress.update(t3, description="[green]ML predictions complete[/green]")

        t4 = progress.add_task("Generating map and charts...", total=None)
        from visualizer import create_forecast_chart, create_risk_chart, create_risk_map
        map_path      = create_risk_map(risk_df)
        chart_path    = create_risk_chart(risk_df)
        forecast_path = create_forecast_chart(risk_df)
        progress.update(t4, description="[green]Outputs saved[/green]")

    console.print()

    from alert_system import run_alerts
    run_alerts(risk_df)

    console.print(f"\n[dim]Interactive map  →  {map_path}[/dim]")
    console.print(f"[dim]Risk chart       →  {chart_path}[/dim]")
    console.print(f"[dim]Forecast chart   →  {forecast_path}[/dim]\n")


if __name__ == "__main__":
    main()
