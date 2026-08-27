import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# ============================================================
# CONFIGURATION & THEME
# ============================================================
st.set_page_config(
    page_title="Kenya Crop Price Monitor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #2E5C3F; }
    .sub-header { font-size: 1.1rem; color: #5A5A5A; margin-bottom: 1rem; }
    .card-container {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFF0E0 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #D4823A;
        margin-bottom: 12px;
    }
    .price-up { color: #C0392B; font-weight: 600; }
    .price-down { color: #27AE60; font-weight: 600; }
    .price-neutral { color: #7F8C8D; font-weight: 600; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-above { background: #FDEDEC; color: #C0392B; }
    .badge-below { background: #EAFAF1; color: #27AE60; }
    .badge-average { background: #FEF9E7; color: #D68910; }
    .insight-box {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #D4823A;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #2C3E50; }
    .metric-label { font-size: 0.8rem; color: #7F8C8D; text-transform: uppercase; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #FFF8F0;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #D4823A !important;
        color: white !important;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E0E0E0;
        font-size: 0.8rem;
        color: #7F8C8D;
        text-align: center;
    }
    .map-legend {
        background: white;
        padding: 10px;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
import os
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'dashboard_data')

@st.cache_data
def load_json(filename):
    with open(os.path.join(data_dir, filename), 'r') as f:
        return json.load(f)

try:
    headlines = load_json('headlines.json')
    drivers = load_json('drivers.json')
    monthly_prices = load_json('monthly_prices.json')
    yearly_prices = load_json('yearly_prices.json')
    equity = load_json('equity.json')
    county_prices = load_json('county_prices.json')
    heatmap_points = load_json('heatmap_points.json')
    with open(os.path.join(data_dir, 'kenya_counties.geojson'), 'r') as f:
        county_geo = json.load(f)
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"Data files not found: {e}")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🌽 Kenya Crop Monitor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Predictions", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("v1.1 • Prototype")

# ============================================================
# PAGE 1: HOME DASHBOARD
# ============================================================
def render_home():
    st.markdown('<p class="main-header">Market Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time price tracking for Kenyan staple crops</p>', unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data files not found. Run the pipeline first.")
        return

    # HEADLINE CARDS
    st.subheader("📊 Crop Price Headlines")
    cols = st.columns(5)
    for i, item in enumerate(headlines):
        with cols[i % 5]:
            arrow = item['trend_arrow']
            change = item['week_change_pct']
            color_class = "price-up" if change > 0 else "price-down" if change < 0 else "price-neutral"
            badge_class = "badge-above" if item['context_badge'] == "Above average" else "badge-below" if item['context_badge'] == "Below average" else "badge-average"
            st.markdown(f"""
            <div class="card-container">
                <div style="font-size: 0.85rem; color: #5A5A5A; margin-bottom: 4px;">{item['commodity']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2C3E50;">KES {item['latest_price']:,.0f}</div>
                <div style="margin: 4px 0;"><span class="{color_class}">{arrow} {abs(change):.1f}%</span></div>
                <div><span class="badge {badge_class}">{item['context_badge']}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # SUMMARY INSIGHTS
    st.subheader("📰 Key Highlights")
    rising = [h for h in headlines if h['week_change_pct'] > 2]
    falling = [h for h in headlines if h['week_change_pct'] < -2]
    drought_crops = [d for d in drivers if d['drought_risk']]
    subsidized = [d for d in drivers if d['subsidy_active']]

    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.markdown('<div class="insight-box"><strong>🌧️ Weather Impact</strong><br>', unsafe_allow_html=True)
        if drought_crops:
            names = ", ".join([d['commodity'] for d in drought_crops[:3]])
            st.markdown(f"Drought conditions for <strong>{names}</strong>. Supply may tighten.", unsafe_allow_html=True)
        else:
            st.markdown("Rainfall patterns within normal ranges.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-box"><strong>💰 Input Costs</strong><br>', unsafe_allow_html=True)
        rising_fert = [d for d in drivers if d['fertilizer_trend'] == 'Rising']
        if rising_fert:
            names = ", ".join([d['commodity'] for d in rising_fert[:3]])
            st.markdown(f"Fertilizer costs rising for <strong>{names}</strong>.", unsafe_allow_html=True)
        else:
            st.markdown("Fertilizer prices stable.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with insight_cols[1]:
        st.markdown('<div class="insight-box"><strong>📈 Price Movers</strong><br>', unsafe_allow_html=True)
        if rising:
            names = ", ".join([h['commodity'] for h in rising[:3]])
            st.markdown(f"<span style='color:#C0392B'>▲ Rising:</span> <strong>{names}</strong>", unsafe_allow_html=True)
        if falling:
            names = ", ".join([h['commodity'] for h in falling[:3]])
            st.markdown(f"<span style='color:#27AE60'>▼ Falling:</span> <strong>{names}</strong>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-box"><strong>🏛️ Policy</strong><br>', unsafe_allow_html=True)
        if subsidized:
            st.markdown("<strong>National Fertilizer Subsidy</strong> active. DAP subsidized ~35%.", unsafe_allow_html=True)
        else:
            st.markdown("No active subsidy programs.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # QUICK STATS
    st.subheader("📋 Quick Stats")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(headlines)}</div><div class="metric-label">Commodities</div></div>', unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{equity["summary"]["total_markets"]}</div><div class="metric-label">Markets</div></div>', unsafe_allow_html=True)
    with stat_cols[2]:
        rising_count = len([h for h in headlines if h['week_change_pct'] > 0])
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#C0392B" if rising_count > len(headlines)/2 else "#27AE60"};">{rising_count}</div><div class="metric-label">Prices Rising</div></div>', unsafe_allow_html=True)
    with stat_cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-value">KES {equity["summary"]["price_range"]["median"]:,.0f}</div><div class="metric-label">Median Price</div></div>', unsafe_allow_html=True)











# ============================================================
# PAGE 5: ABOUT
# ============================================================
def render_about():
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data sources, methodology, and limitations</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 📚 Data Sources

    | Source | Data | Coverage |
    |--------|------|----------|
    | **WFP / HDX** | Monthly food prices by market | 15 Kenyan markets, 10 commodities |
    | **CHIRPS** | Rainfall estimates | Global, 0.05° resolution, 1981–present |
    | **World Bank** | Fertilizer price index | Global monthly, 2010=100 base |
    | **Kenya Met** | Station rainfall (validation) | Major towns |
    | **EPRA** | Fuel prices (future integration) | Monthly pump prices |

    ### ⚠️ Disclaimer

    > **This is a prototype tool for educational and research purposes.** 
    > It is **not** an official government publication. Prices shown are model predictions 
    > based on historical patterns and should not be used as the sole basis for trading 
    > or policy decisions. Always verify with local market reports.

    ### 🧮 Methodology

    **Model:** Gradient Boosting Regressor (scikit-learn)

    **Features:** 61 engineered variables including:
    - Price history (4-week rolling mean, 1–8 week lags)
    - Rainfall (monthly totals, 1–3 month lags, drought/flood flags)
    - Fertilizer costs (DAP/urea prices, subsidy discount, input cost pressure)
    - Seasonal encoding (Kenya's bimodal rainfall cycles)
    - Market type (urban premium, arid transport costs)

    **Validation:** Time-series split (80/20), no shuffle. MAPE: 4.3% on test set.

    ### 🗺️ Map Methodology

    The price heatmap uses:
    - **Choropleth layer**: County polygons colored by selected metric (price, change, inequality)
    - **Circle markers**: Interactive popups with detailed county stats
    - **Heatmap layer**: Intensity-based overlay showing price concentration
    - Counties are simplified polygons for demo; production should use GADM or Kenya Open Data boundaries

    ### 🔒 Limitations

    - **Synthetic data** is used for this demo. Real WFP/CHIRPS data should be substituted for production.
    - **Rainfall-crop causality** is simplified. Actual yield response depends on soil, seed variety, and timing.
    - **Fertilizer prices** are national averages. Regional variations exist.
    - **No conflict/shock modeling.** Sudden events (trade bans, locusts) are not captured.
    - **County boundaries** are approximate. Use official GADM or Kenya Open Data for production.

    ### 🛠️ Technical Stack

    - Python 3.12 + pandas + scikit-learn
    - Streamlit for dashboard
    - Folium + streamlit-folium for maps
    - Data: JSON files (client-side loaded)

    ### 📬 Contact

    For questions or contributions, open an issue on the project repository.
    """)

    with st.expander("🔬 Technical Details: Model Validation"):
        st.markdown("""
        **Model Comparison Results:**

        | Model | MAE (KES) | RMSE (KES) | R² | MAPE |
        |-------|-----------|------------|-----|------|
        | Linear Regression | 304 | 389 | 0.978 | 9.5% |
        | Random Forest (default) | 257 | 346 | 0.982 | 5.6% |
        | Random Forest (deep) | 256 | 343 | 0.983 | 5.6% |
        | Random Forest (wide) | 256 | 343 | 0.983 | 5.6% |
        | Gradient Boosting (default) | 238 | 308 | 0.986 | 5.2% |
        | **Gradient Boosting (tuned)** | **195** | **256** | **0.990** | **4.3%** |

        **Best configuration:** `n_estimators=200, max_depth=5, learning_rate=0.05`

        **Feature importance (top 5):**
        1. `price_roll_mean_4` (90.4%) — 4-week rolling average
        2. `price_change_1w` (3.4%) — Week-over-week momentum
        3. `price_roll_mean_8` (2.1%) — 8-week rolling average
        4. `price_lag_1` (1.5%) — Last week's price
        5. `price_change_4w` (0.8%) — Month-over-month momentum
        """)

with st.expander("📜 License"):
        st.markdown("""
        MIT License — free to use, modify, and distribute with attribution.
        
        Data sources retain their original licenses (WFP Open Data, CHIRPS free for non-commercial).
        """)


# ============================================================
# PAGE 2: PREDICTIONS
# ============================================================
def render_predictions():
    st.markdown('<p class="main-header">📊 Price Predictions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Next-week price predictions for Kenyan staple crops</p>', unsafe_allow_html=True)
    
    if not data_loaded:
        st.error("Data files not found.")
        return
    
    # Check if predictions file exists
    import os
    predictions_path = os.path.join(os.path.dirname(__file__), "kenya_crop_price_project", "kenya_crop_price_predictions_real_data.csv")
    
    if not os.path.exists(predictions_path):
        st.warning("⚠️ Predictions file not found. To generate predictions:")
        st.markdown("""
        1. Ensure you have the required data files in `kenya_crop_price_project/data/`:
           - `wfp_food_prices_kenya.csv` (crop prices from WFP/HDX/Kaggle)
           - The script will generate synthetic rainfall and fertilizer data automatically
        
        2. Run the prediction script:
        ```bash
        cd C:\\Users\\HP\\Downloads\\kcp
        python kenya_crop_price_predictor_real_data.py
        ```
        
        3. Once completed, refresh this page to view the predictions.
        """)
        
        # Show what the predictions would look like
        st.info("""
        The predictions file will contain:
        - Commodity and market pairs
        - Last known price
        - Predicted price for next week
        - Percentage change
        - Rainfall and fertilizer context
        """)
        return
    
    # Load and display predictions
    try:
        predictions_df = pd.read_csv(predictions_path)
        
        st.subheader("📈 Next-Week Price Predictions")
        
        # Format the dataframe for display
        display_df = predictions_df.copy()
        
        # Rename columns for better display if needed
        column_mapping = {
            'commodity': 'Commodity',
            'market': 'Market',
            'last_known_price_kes': 'Last Price (KES)',
            'predicted_price_kes': 'Predicted Price (KES)',
            'change_pct': 'Change (%)',
            'rainfall_mm': 'Rainfall (mm)',
            'dap_price': 'DAP Price (KES)',
            'insights': 'Insights',
            'predicted_date': 'Predicted Date'
        }
        
        # Only rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in display_df.columns}
        display_df = display_df.rename(columns=existing_columns)
        
        # Display the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Add data freshness warning
        from datetime import datetime
        try:
            # Convert predicted_date to datetime for comparison
            display_df['predicted_date_dt'] = pd.to_datetime(display_df['Predicted Date'])
            # Calculate how many days old the data is
            today = datetime.now()
            display_df['days_old'] = (today - display_df['predicted_date_dt']).dt.days
            
            # Check if any data is older than 6 months (approx 180 days)
            stale_data = display_df[display_df['days_old'] > 180]
            if not stale_data.empty:
                st.warning(f"⚠️ {len(stale_data)} predictions are based on data older than 6 months. For most accurate predictions, consider updating the underlying data sources.")
            elif not display_df.empty:
                max_days_old = display_df['days_old'].max()
                if max_days_old > 90:  # Older than 3 months
                    st.info(f"ℹ️ Some predictions are based on data up to {max_days_old} days old. For most recent predictions, consider updating data sources.")
        except Exception as e:
            # If date calculation fails, just continue without the warning
            pass
        
        # Show summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(display_df))
        
        with col2:
            if 'Predicted Price (KES)' in display_df.columns:
                avg_price = display_df['Predicted Price (KES)'].mean()
                st.metric("Avg Predicted Price", f"KES {avg_price:,.0f}")
        
        with col3:
            if 'Change (%)' in display_df.columns:
                avg_change = display_df['Change (%)'].mean()
                st.metric("Avg Change", f"{avg_change:+.1f}%")
        
        with col4:
            if 'Change (%)' in display_df.columns:
                rising_count = len(display_df[display_df['Change (%)'] > 0])
                st.metric("Rising Prices", f"{rising_count}/{len(display_df)}")
        
        # Show insights distribution if available
        if 'Insights' in display_df.columns:
            st.subheader("🔍 Insights Distribution")
            insights_counts = display_df['Insights'].value_counts()
            st.bar_chart(insights_counts)
            
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        st.exception(e)


# ============================================================
# ROUTING
# ============================================================
if page == "🏠 Home":
    render_home()
elif page == "📊 Predictions":
    render_predictions()
elif page == "ℹ️ About":
    render_about()

# Footer
st.markdown("""
<div class="footer">
    Kenya Crop Price Monitor • Prototype v1.1 • Built with Streamlit + Folium
</div>
""", unsafe_allow_html=True)
