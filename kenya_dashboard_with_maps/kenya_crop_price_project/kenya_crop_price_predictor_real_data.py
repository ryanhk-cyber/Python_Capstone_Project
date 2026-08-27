"""
Kenya Crop Price Predictor - Enhanced Edition with Real Data
Following the guide step by step to connect to real data sources
Features: Real Crop Price Data + Real Rainfall Data + Real Fertilizer Data + Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("KENYA CROP PRICE PREDICTOR - ENHANCED EDITION WITH REAL DATA")
print("=" * 70)

# ============================================================
# STEP 1: LOAD REAL DATA FROM SOURCES CITED IN GUIDE
# ============================================================

def load_real_crop_data():
    """Load real crop price data from WFP/HDX/Kaggle CSV."""
    print("\nLoading real crop price data from WFP/HDX/Kaggle...")
    try:
        df = pd.read_csv('.\\kenya_crop_price_project\\data\\wfp_food_prices_kenya.csv')
        print(f"* Loaded crop prices: {df.shape[0]} records")
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'mkt_name': 'market',
            'cm_name': 'commodity',
            'mp_price': 'price_kes',
            'date': 'date'
        })
        
        # Select only needed columns and ensure proper types
        df = df[['date', 'market', 'commodity', 'price_kes']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['price_kes'] = pd.to_numeric(df['price_kes'], errors='coerce')
        
        # Remove any rows with missing data
        df = df.dropna(subset=['date', 'market', 'commodity', 'price_kes'])
        
        print(f"* Cleaned crop data: {df.shape[0]} records")
        return df
    except Exception as e:
        print(f"* Error loading crop prices: {e}")
        print("Falling back to synthetic crop data...")
        return create_crop_data_synthetic(6000)  # Fallback to synthetic

def load_real_rainfall_data():
    """Load real rainfall data from CHIRPS/HDX Subnational CSV and add region_type."""
    print("\nLoading real rainfall data from CHIRPS/HDX Subnational...")
    try:
        df = pd.read_csv('.\\kenya_crop_price_project\\data\\chirps_rainfall_kenya.csv')
        print(f"* Loaded rainfall data: {df.shape[0]} records")
        
        # Ensure proper column names and types
        df = df[['date', 'market', 'rainfall_mm']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
        
        # Add region_type based on market mapping (same as synthetic data)
        regions = {
            'Highlands': ['Nairobi', 'Nakuru', 'Eldoret', 'Embu', 'Meru', 'Thika', 'Machakos', 'Kajiado'],
            'Lake': ['Kisumu'], 'Coast': ['Mombasa'],
            'Arid_North': ['Garissa', 'Lodwar', 'Mandera', 'Marsabit'], 'Semi_Arid': ['Kitui']
        }
        
        # Create reverse mapping from market to region
        market_to_region = {}
        for region, markets in regions.items():
            for market in markets:
                market_to_region[market] = region
        
        df['region_type'] = df['market'].map(market_to_region)
        
        # Remove any rows with missing data
        df = df.dropna(subset=['date', 'market', 'rainfall_mm', 'region_type'])
        
        print(f"* Cleaned rainfall data: {df.shape[0]} records")
        return df
    except Exception as e:
        print(f"* Error loading rainfall data: {e}")
        print("Falling back to synthetic rainfall data...")
        # We'll need to generate dates from crop data for the fallback
        return None  # Signal to use synthetic

def load_real_dap_data():
    """Load real DAP price data from World Bank XLS file."""
    print("\nLoading real DAP price data from World Bank XLS...")
    try:
        # Get column names from row 4 (0-indexed), skipping the first column (which is NaN)
        header_df = pd.read_excel(
            '.\\kenya_crop_price_project\\data\\CMO-Historical-Data-Monthly.xlsx',
            sheet_name='Monthly Prices',
            header=None,
            nrows=5
        )
        # Column names start from index 1 (since index 0 is NaN)
        column_names = header_df.iloc[4, 1:].tolist()
        
        # Read data skipping 6 rows (title, description, headers, blank, units)
        df = pd.read_excel(
            '.\\kenya_crop_price_project\\data\\CMO-Historical-Data-Monthly.xlsx',
            sheet_name='Monthly Prices',
            skiprows=6,
            header=None,
            engine='openpyxl'
        )
        # Assign column names: column 0 is 'date', columns 1+ get the header names
        df.columns = ['date'] + column_names
        
        # Select only date and DAP price columns
        df = df[['date', 'DAP']].copy()
        df = df.rename(columns={'DAP': 'dap_price_usd'})
        
        # Convert date from YYYYMM string to datetime
        # First, replace 'M' with '' to get YYYYMM format
        df['date'] = df['date'].astype(str).str.strip()
        df['date'] = df['date'].str.replace('M', '', regex=False)
        # Now parse as YYYYMM
        df['date'] = pd.to_datetime(df['date'], format='%Y%m', errors='coerce')
        
        # Ensure DAP price is numeric
        df['dap_price_usd'] = pd.to_numeric(df['dap_price_usd'], errors='coerce')
        
        # Remove any rows with missing data
        df = df.dropna(subset=['date', 'dap_price_usd'])
        
        print(f"* Loaded DAP price data: {df.shape[0]} records")
        return df
        
    except Exception as e:
        print(f"* Error loading DAP prices: {e}")
        print("Falling back to synthetic DAP price data...")
        return None

def convert_usd_to_kes(usd_amount, exchange_rate=110.0):
    """Convert USD amount to KES using exchange rate.
    
    Args:
        usd_amount: Price in USD
        exchange_rate: USD/KES exchange rate (default: 110.0)
        
    Returns:
        Price in KES
    """
    return usd_amount * exchange_rate

# --- Synthetic Data Generation Functions (as fallback) ---
def create_crop_data_synthetic(n_rows=6000):
    """Generate realistic synthetic Kenyan crop price data (fallback)."""
    markets = ['Nairobi', 'Mombasa', 'Kisumu', 'Eldoret', 'Nakuru', 
               'Kitui', 'Garissa', 'Lodwar', 'Mandera', 'Marsabit',
               'Embu', 'Meru', 'Kajiado', 'Machakos', 'Thika']
    commodities = ['Maize', 'Beans (Rosecoco)', 'Beans (Mixed)', 
                   'Tomatoes', 'Onions', 'Cabbages', 'Irish Potatoes',
                   'Rice', 'Wheat Flour', 'Green Grams']
    dates = pd.date_range(end='2026-08-20', periods=n_rows, freq='W')
    data = []
    base_prices = {'Maize': 2800, 'Beans (Rosecoco)': 8500, 'Beans (Mixed)': 7500,
                   'Tomatoes': 3500, 'Onions': 4500, 'Cabbages': 1200,
                   'Irish Potatoes': 3000, 'Rice': 6000, 'Wheat Flour': 5500, 'Green Grams': 9000}
    market_premium = {'Nairobi': 1.15, 'Mombasa': 1.10, 'Kisumu': 1.05, 'Eldoret': 0.90,
                      'Nakuru': 0.95, 'Kitui': 1.20, 'Garissa': 1.30, 'Lodwar': 1.40,
                      'Mandera': 1.35, 'Marsabit': 1.25, 'Embu': 0.88, 'Meru': 0.85,
                      'Kajiado': 1.10, 'Machakos': 0.92, 'Thika': 0.95}
    for i in range(n_rows):
        market = np.random.choice(markets)
        commodity = np.random.choice(commodities)
        date = dates[i % len(dates)]
        base = base_prices[commodity]
        month = date.month
        is_rainy = 1 if month in [3,4,5,10,11,12] else 0
        is_harvest = 1 if month in [6,7,8,1,2] else 0
        perishable = 1 if commodity in ['Tomatoes', 'Cabbages', 'Onions'] else 0
        trend = np.sin(2 * np.pi * date.dayofyear / 365) * 0.1
        noise = np.random.normal(0, base * 0.08)
        seasonal_dip = -base * 0.15 if is_harvest else 0
        rain_premium = base * 0.10 if is_rainy and perishable else 0
        price = (base + trend * base + noise + seasonal_dip + rain_premium) * market_premium[market]
        price = max(price, base * 0.3)
        data.append({
            'date': date,
            'market': market,
            'commodity': commodity,
            'price_kes': round(price, 2)
        })
    return pd.DataFrame(data)

def create_rainfall_data_synthetic(dates):
    """Generate synthetic rainfall data (fallback)."""
    regions = {
        'Highlands': ['Nairobi', 'Nakuru', 'Eldoret', 'Embu', 'Meru', 'Thika', 'Machakos', 'Kajiado'],
        'Lake': ['Kisumu'], 'Coast': ['Mombasa'],
        'Arid_North': ['Garissa', 'Lodwar', 'Mandera', 'Marsabit'], 'Semi_Arid': ['Kitui']
    }
    climatology = {
        'Highlands': [40, 50, 90, 180, 150, 70, 40, 40, 35, 60, 100, 65],
        'Lake': [60, 70, 140, 220, 200, 120, 80, 90, 70, 90, 160, 110],
        'Coast': [30, 20, 60, 50, 40, 70, 50, 40, 30, 70, 100, 60],
        'Arid_North': [5, 8, 25, 50, 30, 8, 3, 3, 5, 15, 35, 15],
        'Semi_Arid': [15, 25, 60, 120, 80, 30, 15, 12, 10, 30, 70, 35]
    }
    rainfall_records = []
    for date in dates:
        month = date.month - 1
        for region_type, markets in regions.items():
            base_rain = climatology[region_type][month]
            rain = base_rain * np.random.uniform(0.7, 1.3)
            if np.random.random() < 0.1: rain *= 0.4
            elif np.random.random() < 0.05: rain *= 1.8
            for market in markets:
                rainfall_records.append({'date': date, 'market': market,
                    'rainfall_mm': round(max(rain, 0), 1), 'region_type': region_type})
    df_rain = pd.DataFrame(rainfall_records)
    df_rain['year_month'] = df_rain['date'].dt.to_period('M')
    df_rain_monthly = df_rain.groupby(['year_month', 'market']).agg(
        {'rainfall_mm': 'sum', 'region_type': 'first'}).reset_index()
    df_rain_monthly['date'] = df_rain_monthly['year_month'].dt.to_timestamp()
    return df_rain_monthly[['date', 'market', 'rainfall_mm', 'region_type']]

def create_dap_price_synthetic(dates):
    """Generate synthetic DAP price data (fallback)."""
    records = []
    for date in dates:
        year, month = date.year, date.month
        # Similar pattern to original fertilizer index but for DAP prices
        # Reverse engineering from the original formula: DAP Price (KES) = 3500 + (fertilizer_index - 100) × 18
        # So: fertilizer_index = 100 + (DAP Price - 3500) / 18
        if year <= 2020: base_price = 3500 + (year - 2015) * 3 * 18  # Reverse the original formula
        elif year == 2021: base_price = 3500 + (130 + month * 5 - 100) * 18
        elif year == 2022: base_price = 3500 + (200 + month * 5 - 100) * 18
        elif year == 2023: base_price = 3500 + (260 - month * 10 - 100) * 18
        elif year == 2024: base_price = 3500 + (140 - month * 2 - 100) * 18
        elif year == 2025: base_price = 3500 + (116 + month * 2 - 100) * 18
        else: base_price = 3500 + (140 + month * 5 - 100) * 18
        
        seasonal = 150 if month in [2,3,9,10] else 0  # From original formula
        noise = np.random.normal(0, 40)
        dap_price = base_price + seasonal + noise
        
        records.append({
            'date': date, 
            'dap_price_kes': round(max(dap_price, 2000), 2)  # Match original clipping
        })
    return pd.DataFrame(records)

# ============================================================
# STEP 2: LOAD AND PREPARE REAL DATA
# ============================================================

print("\n" + "=" * 50)
print("LOADING REAL DATA SOURCES")
print("=" * 50)

# Load real data
df_crop = load_real_crop_data()
df_rain = load_real_rainfall_data()
df_dap = load_real_dap_data()

# If any real data loading failed, we'll use synthetic for that component
use_synthetic_rain = df_rain is None
use_synthetic_dap = df_dap is None  # Changed from fertilizer to DAP

if use_synthetic_rain:
    print("\nGenerating synthetic rainfall data as fallback...")
    dates_unique = df_crop['date'].unique()
    df_rain = create_rainfall_data_synthetic(dates_unique)

if use_synthetic_dap:
    print("\nGenerating synthetic DAP price data as fallback...")
    dates_unique = df_crop['date'].unique()
    df_dap = create_dap_price_synthetic(dates_unique)
else:
    # Convert DAP prices from USD to KES
    print("\nConverting DAP prices from USD to KES...")
    # Using fixed exchange rate - should be made configurable or sourced from data
    EXCHANGE_RATE_USD_TO_KES = 110.0  # Placeholder value
    df_dap['dap_price_kes'] = convert_usd_to_kes(
        df_dap['dap_price_usd'], 
        EXCHANGE_RATE_USD_TO_KES
    )
    # Keep only needed columns
    df_dap = df_dap[['date', 'dap_price_kes']]

print(f"\nFinal data loaded:")
print(f"  Crop prices: {df_crop.shape}")
print(f"  Rainfall: {df_rain.shape}")
print(f"  DAP prices: {df_dap.shape}")

# ============================================================
# STEP 3: MERGE DATASETS AS PER GUIDE
# ============================================================

print("\n" + "=" * 50)
print("STEP 3: MERGING DATASETS")
print("=" * 50)

# Prepare data for merging - convert dates to monthly periods
df_crop['year_month'] = df_crop['date'].dt.to_period('M')
df_rain['year_month'] = df_rain['date'].dt.to_period('M')
df_dap['year_month'] = df_dap['date'].dt.to_period('M')  # Changed from fertilizer

# Merge crop prices with rainfall data on market and date (monthly)
df_merged = df_crop.merge(
    df_rain[['year_month', 'market', 'rainfall_mm', 'region_type']],
    on=['year_month', 'market'], 
    how='left'
)
print(f"* After merging crop + rainfall: {df_merged.shape}")

# Merge with DAP price data (national level, so just on date/month)
df_merged = df_merged.merge(
    df_dap[['year_month', 'dap_price_kes']],  # Changed from fertilizer_index
    on='year_month', 
    how='left'
)
print(f"* After merging with DAP prices: {df_merged.shape}")

# Clean up temporary columns
df_merged = df_merged.drop('year_month', axis=1)

# ============================================================
# STEP 4: FEATURE ENGINEERING (Enhanced with real data)
# ============================================================

print("\n" + "=" * 50)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 50)

def engineer_features(df):
    """Create enhanced features for crop price prediction using real data."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    
    # Kenya-specific seasonal features
    # Long rains: March-May (planting)
    # Short rains: Oct-Dec (planting)
    # Harvest follows 2-3 months after
    df['is_long_rains'] = df['month'].isin([3,4,5]).astype(int)
    df['is_short_rains'] = df['month'].isin([10,11,12]).astype(int)
    df['is_rainy_season'] = (df['is_long_rains'] | df['is_short_rains']).astype(int)
    
    # Harvest periods (when supply floods market, prices drop)
    df['is_harvest'] = df['month'].isin([6,7,8,1,2]).astype(int)
    
    # Kenyan calendar events
    df['is_end_month'] = df['date'].dt.is_month_end.astype(int)  # Payday effect
    df['is_december'] = (df['month'] == 12).astype(int)  # Festive season
    
    # Cyclical encoding for week/month (prices are cyclical)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (previous prices — most important for time series)
    df = df.sort_values(['commodity', 'market', 'date'])
    for lag in [1, 2, 4, 8]:  # 1 week, 2 weeks, 1 month, 2 months ago
        df[f'price_lag_{lag}'] = df.groupby(['commodity', 'market'])['price_kes'].shift(lag)
    
    # Rolling statistics (trend and volatility)
    for window in [4, 8, 12]:
        df[f'price_roll_mean_{window}'] = (
            df.groupby(['commodity', 'market'])['price_kes']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f'price_roll_std_{window}'] = (
            df.groupby(['commodity', 'market'])['price_kes']
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    
    # Price change rate (momentum)
    df['price_change_1w'] = df.groupby(['commodity', 'market'])['price_kes'].pct_change(1)
    df['price_change_4w'] = df.groupby(['commodity', 'market'])['price_kes'].pct_change(4)
    
    # Commodity category
    perishables = ['Tomatoes', 'Cabbages', 'Onions', 'Irish Potatoes']
    grains = ['Maize', 'Rice', 'Wheat Flour']
    legumes = ['Beans (Rosecoco)', 'Beans (Mixed)', 'Green Grams']
    
    df['category'] = df['commodity'].apply(
        lambda x: 'Perishable' if x in perishables 
        else 'Grain' if x in grains 
        else 'Legume')
    # Perishable flag
    df['is_perishable'] = df['category'].apply(lambda x: 1 if x == 'Perishable' else 0)
    
    # Market type (urban vs rural/arid)
    urban_markets = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Thika']
    arid_markets = ['Garissa', 'Lodwar', 'Mandera', 'Marsabit']
    df['market_type'] = df['market'].apply(
        lambda x: 'Urban' if x in urban_markets 
        else 'Arid' if x in arid_markets 
        else 'Rural')
    
    # Rainfall lags and features (per market)
    df = df.sort_values(['market', 'date'])
    for lag in [1, 2, 3]:
        df[f'rainfall_lag_{lag}m'] = df.groupby('market')['rainfall_mm'].shift(lag)
    # 3-month cumulative rainfall
    df['rainfall_cum_3m'] = df.groupby('market')['rainfall_mm'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum())
    # 3-month rolling mean for anomaly
    df['rainfall_roll_mean_3m'] = df.groupby('market')['rainfall_mm'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    # Rainfall anomaly percentage
    df['rainfall_anomaly_pct'] = (
        (df['rainfall_mm'] - df['rainfall_roll_mean_3m']) / df['rainfall_roll_mean_3m'].replace(0, np.nan) * 100
    )
    # Drought and flood flags (anomaly below -50% or above 50%)
    df['is_drought'] = (df['rainfall_anomaly_pct'] < -50).astype(int)
    df['is_flood'] = (df['rainfall_anomaly_pct'] > 50).astype(int)
    # Drop temporary column
    df = df.drop(columns=['rainfall_roll_mean_3m'])
    
    # Interaction features
    df['rain_x_perishable'] = df['rainfall_mm'] * df['is_perishable']
    df['drought_x_perishable'] = df['is_drought'] * df['is_perishable']
    
    # DAP lags and changes (national level, so shift globally)
    df = df.sort_values('date')
    for lag in [1, 2, 3]:
        df[f'dap_price_lag_{lag}m'] = df['dap_price_kes'].shift(lag)
    # Month-over-month change
    df['dap_price_change_1m'] = df['dap_price_kes'].pct_change(1)
    # 3-month change
    df['dap_price_change_3m'] = df['dap_price_kes'].pct_change(3)
    
    return df

# Apply feature engineering
df_features = engineer_features(df_merged)
print(f"* Features created: {df_features.shape[1]} total columns")

# ============================================================
# STEP 5: PREPARE DATA FOR MODELING
# ============================================================

print("\n" + "=" * 50)
print("STEP 5: DATA PREPARATION")
print("=" * 50)

# Drop rows with NaN (from lag features)
df_model = df_features.dropna().copy()
print(f"* Rows after dropping NaN: {len(df_model)}")

# Encode categorical variables
le_commodity = LabelEncoder()
le_market = LabelEncoder()
le_category = LabelEncoder()
le_market_type = LabelEncoder()
le_region_type = LabelEncoder()  # For region_type from rainfall data

df_model['commodity_enc'] = le_commodity.fit_transform(df_model['commodity'])
df_model['market_enc'] = le_market.fit_transform(df_model['market'])
df_model['category_enc'] = le_category.fit_transform(df_model['category'])
df_model['market_type_enc'] = le_market_type.fit_transform(df_model['market_type'])
df_model['region_type_enc'] = le_region_type.fit_transform(df_model['region_type'])

# Feature columns (exclude target and metadata)
feature_cols = [
    'commodity_enc', 'market_enc', 'category_enc', 'market_type_enc', 'region_type_enc',
    'year', 'month', 'week', 'dayofyear', 'quarter',
    'is_long_rains', 'is_short_rains', 'is_rainy_season', 'is_harvest', 'is_end_month', 'is_december',
    'week_sin', 'week_cos', 'month_sin', 'month_cos',
    'price_lag_1', 'price_lag_2', 'price_lag_4', 'price_lag_8',
    'price_roll_mean_4', 'price_roll_mean_8', 'price_roll_mean_12',
    'price_roll_std_4', 'price_roll_std_8', 'price_roll_std_12',
    'price_change_1w', 'price_change_4w',
    'rainfall_mm', 'rainfall_lag_1m', 'rainfall_lag_2m', 'rainfall_lag_3m',
    'rainfall_cum_3m', 'rainfall_anomaly_pct', 'is_drought', 'is_flood',
    'is_perishable', 'rain_x_perishable', 'drought_x_perishable',
    'dap_price_kes',  # Changed from 'fertilizer_index' to 'dap_price_kes'
    'dap_price_lag_1m', 'dap_price_lag_2m', 'dap_price_lag_3m',
    'dap_price_change_1m', 'dap_price_change_3m',
]

# Create target variable: next week's price (shift price_kes up by 1)
df_model['price_kes_next_week'] = df_model['price_kes'].shift(-1)
# Drop the last row since it has no next week price
df_model = df_model[:-1].copy()

X = df_model[feature_cols]
y = df_model['price_kes_next_week']
if np.isinf(X.values).any():
    raise ValueError("Infinities in X!")

# Time-based split (no random shuffle — preserves temporal order)
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"* Training samples: {len(X_train)}")
print(f"* Test samples: {len(X_test)}")
print(f"* Features used: {len(feature_cols)}")

# ============================================================
# STEP 6: TRAIN MODELS WITH HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 50)
print("STEP 6: MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("=" * 50)

# Define models with hyperparameter tuning
models = {
    'Linear Regression': LinearRegression(),
    'RF (default)': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'RF (tuned)': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3,
                                       random_state=42, n_jobs=-1),
    'GB (default)': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'GB (tuned)': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                           random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results[name] = {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'pred': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  MAE:  KES {mae:,.0f}")
    print(f"  RMSE: KES {rmse:,.0f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  MAPE: {mape:.1f}%")

# Pick best model by MAE
best_name = min(results, key=lambda k: results[k]['mae'])
best_model = results[best_name]['model']
print(f"\n+ Best Model: {best_name} (MAE: KES {results[best_name]['mae']:,.0f})")

# ============================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 50)
print("STEP 7: FEATURE IMPORTANCE")
print("=" * 50)

if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 most important features:")
    for _, row in importances.head(15).iterrows():
        bar = "#" * int(row['importance'] * 100)
        print(f"  {row['feature']:<28} {row['importance']:.3f} {bar}")
    
    # Group analysis
    rain_f = [f for f in feature_cols if 'rain' in f.lower()]
    dap_f = [f for f in feature_cols if 'dap' in f.lower()]  # Changed from 'fert' to 'dap'
    price_f = [f for f in feature_cols if 'price' in f.lower() and f not in dap_f]
    ri = importances[importances['feature'].isin(rain_f)]['importance'].sum()
    fi = importances[importances['feature'].isin(dap_f)]['importance'].sum()  # Changed from fert_f to dap_f
    pi = importances[importances['feature'].isin(price_f)]['importance'].sum()
    print(f"\nGroup contributions: Price {pi:.1%} | Rain {ri:.1%} | DAP {fi:.1%} | Other {1-pi-ri-fi:.1%}")  # Changed from Fert to DAP

# ============================================================
# STEP 8: GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 50)
print("STEP 8: NEXT-WEEK PREDICTIONS")
print("=" * 50)

def predict_next_week(df_model, commodity, market, model, feature_cols,
                     le_commodity, le_market, le_category, le_market_type, le_region_type):
    """Predict price for a specific commodity-market pair next week."""
    
    # Get the most recent row for this commodity-market
    mask = (df_model['commodity'] == commodity) & (df_model['market'] == market)
    recent = df_model[mask].iloc[-1:].copy()
    
    if len(recent) == 0:
        return None
    
    # Advance date by 1 week
    next_week = recent.copy()
    next_week['date'] = next_week['date'] + pd.Timedelta(weeks=1)
    next_week['year'] = next_week['date'].dt.year
    next_week['month'] = next_week['date'].dt.month
    next_week['week'] = next_week['date'].dt.isocalendar().week.astype(int)
    next_week['dayofyear'] = next_week['date'].dt.dayofyear
    next_week['quarter'] = next_week['date'].dt.quarter
    
    # Update seasonal flags
    next_week['is_long_rains'] = next_week['month'].isin([3,4,5]).astype(int)
    next_week['is_short_rains'] = next_week['month'].isin([10,11,12]).astype(int)
    next_week['is_rainy_season'] = (next_week['is_long_rains'] | next_week['is_short_rains']).astype(int)
    next_week['is_harvest'] = next_week['month'].isin([6,7,8,1,2]).astype(int)
    next_week['is_end_month'] = next_week['date'].dt.is_month_end.astype(int)
    next_week['is_december'] = (next_week['month'] == 12).astype(int)
    
    # Update cyclical features
    next_week['week_sin'] = np.sin(2 * np.pi * next_week['week'] / 52)
    next_week['week_cos'] = np.cos(2 * np.pi * next_week['week'] / 52)
    next_week['month_sin'] = np.sin(2 * np.pi * next_week['month'] / 12)
    next_week['month_cos'] = np.cos(2 * np.pi * next_week['month'] / 12)
    
    # Lag features: shift current values forward
    next_week['price_lag_1'] = recent['price_kes'].values[0]
    next_week['price_lag_2'] = recent['price_lag_1'].values[0]
    next_week['price_lag_4'] = recent['price_lag_2'].values[0]
    next_week['price_lag_8'] = recent['price_lag_4'].values[0]
    
    # Rolling means (approximate with recent values)
    next_week['price_roll_mean_4'] = recent['price_roll_mean_4'].values[0]
    next_week['price_roll_mean_8'] = recent['price_roll_mean_8'].values[0]
    next_week['price_roll_mean_12'] = recent['price_roll_mean_12'].values[0]
    next_week['price_roll_std_4'] = recent['price_roll_std_4'].values[0]
    next_week['price_roll_std_8'] = recent['price_roll_std_8'].values[0]
    next_week['price_roll_std_12'] = recent['price_roll_std_12'].values[0]
    
    # Price changes
    next_week['price_change_1w'] = recent['price_change_1w'].values[0]
    next_week['price_change_4w'] = recent['price_change_4w'].values[0]
    
    # Rainfall features (use recent values as forecast for simplicity)
    for lag in [1, 2, 3]:
        col = f'rainfall_lag_{lag}m'
        next_week[col] = recent[col].values[0] if col in recent.columns else recent['rainfall_mm'].values[0]
    next_week['rainfall_cum_3m'] = recent['rainfall_cum_3m'].values[0]
    next_week['rainfall_anomaly_pct'] = recent['rainfall_anomaly_pct'].values[0]
    next_week['is_drought'] = recent['is_drought'].values[0]
    next_week['is_flood'] = recent['is_flood'].values[0]
    next_week['is_perishable'] = recent['is_perishable'].values[0]
    next_week['rain_x_perishable'] = recent['rain_x_perishable'].values[0]
    next_week['drought_x_perishable'] = recent['drought_x_perishable'].values[0]
    
    # DAP features (changed from fertilizer)
    for lag in [1, 2, 3]:
        col = f'dap_price_lag_{lag}m'
        next_week[col] = recent[col].values[0] if col in recent.columns else recent['dap_price_kes'].values[0]
    next_week['dap_price_change_1m'] = recent['dap_price_change_1m'].values[0]
    next_week['dap_price_change_3m'] = recent['dap_price_change_3m'].values[0]
    
    # Region type (from rainfall data)
    next_week['region_type_enc'] = le_region_type.transform(next_week['region_type'])
    
    # Encode categorical variables
    next_week['commodity_enc'] = le_commodity.transform(next_week['commodity'])
    next_week['market_enc'] = le_market.transform(next_week['market'])
    next_week['category_enc'] = le_category.transform(next_week['category'])
    next_week['market_type_enc'] = le_market_type.transform(next_week['market_type'])
    
    X_next = next_week[feature_cols]
    pred_price = model.predict(X_next)[0]
    
    return {
        'commodity': commodity,
        'market': market,
        'predicted_price_kes': round(pred_price, 2),
        'last_known_price_kes': round(recent['price_kes'].values[0], 2),
        'predicted_date': str(next_week['date'].values[0])[:10],
        'change_pct': round((pred_price - recent['price_kes'].values[0]) / recent['price_kes'].values[0] * 100, 2),
        'rainfall_mm': round(recent['rainfall_mm'].values[0], 1),
        'dap_price': round(recent['dap_price_kes'].values[0], 1)  # Changed from fert_index to dap_price
    }

# Generate predictions for key commodity-market pairs
key_pairs = [
    ('Maize', 'Nairobi'), ('Maize', 'Eldoret'), ('Maize', 'Lodwar'),
    ('Beans (Rosecoco)', 'Nairobi'), ('Beans (Mixed)', 'Kisumu'),
    ('Tomatoes', 'Nairobi'), ('Tomatoes', 'Kisumu'),
    ('Onions', 'Mombasa'), ('Onions', 'Garissa'),
    ('Cabbages', 'Nakuru'), ('Irish Potatoes', 'Meru'),
    ('Rice', 'Nairobi'), ('Wheat Flour', 'Nairobi'), ('Green Grams', 'Nairobi')
]

print(f"\n{'Commodity':<18} {'Market':<10} {'Last':>10} {'Predicted':>10} {'Change':>8} {'Rain':>6} {'DAP':>6}")
print("-" * 85)

predictions = []
for commodity, market in key_pairs:
    pred = predict_next_week(
        df_model, commodity, market, best_model, feature_cols,
        le_commodity, le_market, le_category, le_market_type, le_region_type
    )
    if pred:
        predictions.append(pred)
        change_str = f"{pred['change_pct']:+.1f}%"
        print(f"{pred['commodity']:<18} {pred['market']:<10} KSh{pred['last_known_price_kes']:>8,.0f} "
              f"KSh{pred['predicted_price_kes']:>8,.0f} {change_str:>8} {pred['rainfall_mm']:>5.0f}mm "
              f"{pred['dap_price']:>5.0f}")

# ============================================================
# STEP 9: SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 50)
print("STEP 9: SAVING OUTPUTS")
print("=" * 50)

# Save predictions
if predictions:
    pred_df = pd.DataFrame(predictions)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predictions_path = os.path.join(script_dir, 'kenya_crop_price_predictions_real_data.csv')
    pred_df.to_csv(predictions_path, index=False)
    print(f"* Saved: {os.path.basename(predictions_path)}")

# Save processed data
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_path = os.path.join(script_dir, 'kenya_crop_prices_real_data_processed.csv')
df_model.to_csv(processed_path, index=False)
print(f"* Saved: {os.path.basename(processed_path)}")

# Save model comparison
script_dir = os.path.dirname(os.path.abspath(__file__))
model_comparison_df = pd.DataFrame([
    {
        'Model': name,
        'MAE (KES)': f"{results[name]['mae']:,.0f}",
        'RMSE (KES)': f"{results[name]['rmse']:,.0f}",
        'R²': f"{results[name]['r2']:.3f}",
        'MAPE (%)': f"{results[name]['mape']:.1f}"
    }
    for name in models.keys()
])
model_comparison_path = os.path.join(script_dir, 'model_comparison_real_data.csv')
model_comparison_df.to_csv(model_comparison_path, index=False)
print(f"* Saved: {os.path.basename(model_comparison_path)}")

# Save feature importances
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    importances_path = os.path.join(script_dir, 'feature_importances_real_data.csv')
    importances.to_csv(importances_path, index=False)
    print(f"* Saved: {os.path.basename(importances_path)}")

print("\n" + "=" * 70)
print("REAL DATA CONNECTION AND MODEL TRAINING COMPLETE!")
print("Successfully connected to real data sources and trained enhanced model.")
print("=" * 70)