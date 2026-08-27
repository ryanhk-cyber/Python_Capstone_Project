# 🌽 Kenya Crop Price Predictor

ML model predicting next-week prices for Kenyan staples (maize, beans, tomatoes)
across 15 markets, incorporating rainfall and fertilizer data.

## Problem
Kenyan staple prices swing wildly due to seasonal rains, harvest floods, and
input cost shocks. Farmers and traders need early price signals.

## Solution
- **61 engineered features** from 3 data sources
- **Gradient Boosting** with hyperparameter tuning (MAE: KES 195, MAPE: 4.3%)
- **Contextual predictions**: drought alerts, fertilizer cost pressure, subsidy status

## Data Sources
| Source | Data | Link |
|--------|------|------|
| WFP/HDX | Crop prices by market | [HDX Kenya](https://data.humdata.org/dataset/wfp-food-prices-for-kenya) |
| CHIRPS | Monthly rainfall | [CHIRPS](https://data.chc.ucsb.edu/products/CHIRPS-2.0/) |
| World Bank | Fertilizer price index | [WB Commodities](https://www.worldbank.org/en/research/commodity-markets) |

## Key Features
- **Rainfall lags**: 1–3 month lagged rainfall (affects harvest 6–12 weeks later)
- **Drought/flood flags**: Anomaly detection vs. long-term monthly averages
- **Fertilizer cost pressure**: DAP price × grain flag (grains are fertilizer-intensive)
- **Subsidy discount**: Kenya National Fertilizer Subsidy impact since 2022

## Results
| Model | MAE (KES) | MAPE |
|-------|-----------|------|
| Linear Regression | 304 | 9.5% |
| Random Forest | 257 | 5.6% |
| **Gradient Boosting (tuned)** | **195** | **4.3%** |

## Run Locally
```bash
pip install -r requirements.txt
python kenya_crop_price_predictor.py
streamlit run app.py
```

## Live Demo
[Streamlit App](https://your-app-url.streamlit.app)