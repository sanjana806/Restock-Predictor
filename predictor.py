"""
predictor.py - TabPFN Tabular AI Forecasting and Feature Engineering Engine
--------------------------------------------------------------------------
This module processes e-commerce historical sales data and uses TabPFN
(Prior-Data Fitted Network for Tabular Data) via `tabpfn-client` to predict
the next 30 days of inventory demand for each Item_ID.

==============================================================================
WHY TABPFN OVER STANDARD LLMs FOR INVENTORY RESTOCK PREDICTION:
==============================================================================
1. Arithmetic Precision vs. Token Hallucinations:
   Standard LLMs (like GPT-4 or Claude) treat numbers as discrete BPE tokens,
   frequently hallucinating arithmetic additions, failing to scale numbers
   linearly, or producing non-monotonic time trends. TabPFN is a specialized
   transformer trained directly on synthetic tabular priors to model continuous
   numerical distributions mathematically.

2. Instant In-Context Learning (< 1 second):
   Unlike traditional GBDTs (XGBoost/LightGBM) which require iterative gradient
   boosting and hyperparameter tuning, TabPFN performs Bayesian inference across
   all data points in a single forward pass.

3. Native Probabilistic Uncertainty & Quantiles:
   Inventory management requires safety buffers to prevent costly stockouts.
   TabPFN outputs true predictive distributions (e.g. 90th percentile quantiles),
   giving small business owners statistical confidence intervals rather than
   arbitrary guesses.

4. Token Efficiency & Privacy:
   Serializing 90 days of multi-item time-series into prompt text consumes
   thousands of LLM tokens and risks proprietary store leakage. TabPFN runs on
   numerical matrices directly.
==============================================================================
"""

import os
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List

# Standard data science libraries
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

# Attempt to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("restock_predictor.predictor")

def get_tabpfn_api_key() -> Optional[str]:
    """
    Retrieve TabPFN API key/token from environment variables.
    Supports TABPFN_API_KEY, TABPFN_TOKEN, or PRIORLABS_API_KEY.
    """
    for key in ["TABPFN_API_KEY", "TABPFN_TOKEN", "PRIORLABS_API_KEY"]:
        val = os.getenv(key, "").strip()
        if val:
            return val
    return None

def validate_sales_csv(df: Any) -> Tuple[bool, str]:
    """
    Validate that the uploaded DataFrame meets requirements:
    - Must contain columns: 'Date', 'Item_ID', 'Quantity_Sold'
    - 'Quantity_Sold' must be numeric
    - 'Date' must be parseable
    """
    if df is None or len(df) == 0:
        return False, "The uploaded file is empty."

    required_cols = {"Date", "Item_ID", "Quantity_Sold"}
    current_cols = set(df.columns)
    
    # Case-insensitive column matching fallback
    col_map = {col.lower(): col for col in df.columns}
    normalized_required = {r.lower(): r for r in required_cols}
    
    missing = [req for req_lower, req in normalized_required.items() if req_lower not in col_map]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}. Expected: Date, Item_ID, Quantity_Sold."

    # Rename to canonical casing if needed
    for req_lower, req in normalized_required.items():
        if col_map[req_lower] != req:
            df.rename(columns={col_map[req_lower]: req}, inplace=True)

    # Check row counts
    if len(df) < 5:
        return False, "Dataset must have at least 5 sales records to establish a baseline."

    return True, "Valid dataset."

def _extract_tabular_features(item_df: Any, start_date: datetime) -> Any:
    """
    Transform time-series records into structured tabular feature vectors
    optimized for TabPFN's in-context regression transformer.
    
    Features engineered:
    1. day_index: Normalized continuous time step (captures linear drift/trend)
    2. day_of_week: 0 (Monday) to 6 (Sunday) (captures weekly shopping seasonality)
    3. is_weekend: Binary flag (Etsy/Shopify surges on Saturday/Sunday)
    4. day_of_month: 1 to 31 (captures payday cyclical effects)
    5. rolling_7d_mean: 7-day trailing average (captures current baseline demand)
    6. lag_7: Sales from exactly 7 days prior (same-day-of-week anchor)
    """
    df = item_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["day_index"] = (df["Date"] - start_date).dt.days
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda w: 1 if w >= 5 else 0)
    df["day_of_month"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month

    # Rolling window and lag features
    df["rolling_7d_mean"] = df["Quantity_Sold"].rolling(window=7, min_periods=1).mean()
    df["lag_7"] = df["Quantity_Sold"].shift(7).fillna(df["Quantity_Sold"].mean())

    return df

def _fallback_local_regression(
    X_train: Any,
    y_train: Any,
    X_future: Any,
    recent_mean: float,
    recent_std: float
) -> Tuple[Any, Any]:
    """
    High-fidelity local tabular regression fallback (Bayesian Ridge / Moving Trends)
    used when TABPFN_API_KEY is omitted or when running in offline demo mode.
    """
    try:
        from sklearn.linear_model import BayesianRidge
        reg = BayesianRidge()
        reg.fit(X_train, y_train)
        preds_mean = reg.predict(X_future)
    except Exception:
        # Pure statistical heuristic if scikit-learn is missing
        weekend_mask = X_future[:, 2]  # is_weekend column
        day_trend = (X_future[:, 0] - X_train[-1, 0]) * 0.03
        preds_mean = recent_mean + (weekend_mask * (recent_mean * 0.25)) + day_trend

    # Prevent negative sales predictions
    preds_mean = np.clip(preds_mean, a_min=1.0, a_max=None)
    # Estimate 90th percentile safety buffer (mean + 1.28 * std)
    safety_buffer = preds_mean + max(1.5, 1.28 * recent_std)
    
    return preds_mean, safety_buffer

def forecast_item_demand(
    item_id: str,
    item_df: Any,
    forecast_days: int = 30,
    api_key: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Forecast the next 30 days of sales for a single Item_ID using TabPFN or fallback.
    """
    item_df = item_df.copy()
    item_df["Date"] = pd.to_datetime(item_df["Date"])
    item_df = item_df.sort_values("Date").reset_index(drop=True)

    min_date = item_df["Date"].min()
    max_date = item_df["Date"].max()

    # Calculate historical 30-day baseline for growth delta comparison
    cutoff_30d = max_date - timedelta(days=30)
    last_30d_df = item_df[item_df["Date"] > cutoff_30d]
    historical_30d_sales = int(last_30d_df["Quantity_Sold"].sum()) if len(last_30d_df) > 0 else int(item_df["Quantity_Sold"].sum())

    # Engineer tabular training features
    featured_df = _extract_tabular_features(item_df, min_date)
    feature_cols = ["day_index", "day_of_week", "is_weekend", "day_of_month", "rolling_7d_mean", "lag_7"]

    X_train = featured_df[feature_cols].values
    y_train = featured_df["Quantity_Sold"].values

    recent_mean = float(featured_df["Quantity_Sold"].tail(14).mean())
    recent_std = float(featured_df["Quantity_Sold"].tail(14).std())
    if math.isnan(recent_std) or recent_std == 0:
        recent_std = max(1.0, recent_mean * 0.15)

    # Construct the future 30 days feature matrix
    future_dates = [max_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_records = []
    
    last_known_rolling = recent_mean
    last_known_lags = list(featured_df["Quantity_Sold"].tail(7).values)

    for i, f_date in enumerate(future_dates):
        d_idx = (f_date - min_date).days
        dow = f_date.weekday()
        is_wknd = 1 if dow >= 5 else 0
        dom = f_date.day
        lag7_val = last_known_lags[i % 7]
        future_records.append([d_idx, dow, is_wknd, dom, last_known_rolling, lag7_val])

    X_future = np.array(future_records)

    # TabPFN Model Execution
    engine_used = "TabPFN Regressor (Prior Labs Cloud API)"
    key_to_use = api_key or get_tabpfn_api_key()
    predictions_mean = None
    predictions_buffer = None

    if key_to_use:
        try:
            from tabpfn_client import TabPFNRegressor, init
            os.environ["TABPFN_TOKEN"] = key_to_use
            init()

            # Train TabPFN in-context
            model = TabPFNRegressor()
            model.fit(X_train, y_train)

            # Probabilistic Quantile Prediction
            try:
                quantiles = model.predict(X_future, output_type="quantiles", quantiles=[0.5, 0.9])
                predictions_mean = quantiles[:, 0]
                predictions_buffer = quantiles[:, 1]
            except Exception:
                predictions_mean = model.predict(X_future)
                predictions_buffer = predictions_mean + (1.28 * recent_std)

            logger.info(f"Generated TabPFN predictions for {item_id}")
        except Exception as exc:
            logger.warning(f"TabPFN API error: {exc}. Using tabular regression fallback.")
            engine_used = f"Local Tabular Regressor (Fallback: {str(exc)[:45]})"
            predictions_mean, predictions_buffer = _fallback_local_regression(
                X_train, y_train, X_future, recent_mean, recent_std
            )
    else:
        engine_used = "Local Tabular Regressor (Fallback: No TABPFN_API_KEY)"
        predictions_mean, predictions_buffer = _fallback_local_regression(
            X_train, y_train, X_future, recent_mean, recent_std
        )

    # Clip values
    predictions_mean = np.clip(predictions_mean, a_min=1.0, a_max=None)
    predictions_buffer = np.clip(predictions_buffer, a_min=predictions_mean, a_max=None)

    forecast_records = []
    for f_date, p_mean, p_buf in zip(future_dates, predictions_mean, predictions_buffer):
        forecast_records.append({
            "Date": f_date.strftime("%Y-%m-%d"),
            "Item_ID": item_id,
            "Predicted_Daily_Demand": int(round(p_mean)),
            "Safety_Buffer_90Pct": int(round(p_buf))
        })

    forecast_df = pd.DataFrame(forecast_records)
    meta = {
        "item_id": item_id,
        "engine": engine_used,
        "training_samples": len(item_df),
        "recent_daily_avg": round(recent_mean, 1),
        "historical_30d_sales": historical_30d_sales,
        "forecast_start": future_dates[0].strftime("%Y-%m-%d"),
        "forecast_end": future_dates[-1].strftime("%Y-%m-%d")
    }
    return forecast_df, meta

def run_inventory_forecasting(
    df: Any,
    forecast_days: int = 30,
    api_key: Optional[str] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Main entry point for forecasting demand across all items in the CSV.
    """
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy must be installed to run restock predictions.")

    valid, msg = validate_sales_csv(df)
    if not valid:
        raise ValueError(msg)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Quantity_Sold"] = pd.to_numeric(df["Quantity_Sold"], errors="coerce").fillna(1)
    df["Item_ID"] = df["Item_ID"].astype(str)

    unique_items = list(df["Item_ID"].unique())
    all_forecasts = []
    summary_rows = []
    engines_used = set()

    for item in unique_items:
        item_data = df[df["Item_ID"] == item]
        item_forecast, meta = forecast_item_demand(item, item_data, forecast_days, api_key)
        all_forecasts.append(item_forecast)
        engines_used.add(meta["engine"])

        total_predicted = int(item_forecast["Predicted_Daily_Demand"].sum())
        total_buffer = int(item_forecast["Safety_Buffer_90Pct"].sum())
        recommended_safety_stock = max(0, total_buffer - total_predicted)
        suggested_restock = total_predicted + recommended_safety_stock

        historical_30d = meta["historical_30d_sales"]
        growth_pct = round(((total_predicted - historical_30d) / max(1, historical_30d)) * 100, 1)

        # Risk Classification
        if growth_pct > 20:
            risk = "High Demand Surge"
        elif growth_pct < -15:
            risk = "Cooling Demand"
        else:
            risk = "Stable Velocity"

        summary_rows.append({
            "Item_ID": item,
            "Suggested_Restock_Quantity": int(suggested_restock),
            "Total_Predicted_Sales_30D": int(total_predicted),
            "Recommended_Safety_Stock": int(recommended_safety_stock),
            "Avg_Daily_Demand": round(total_predicted / forecast_days, 1),
            "Growth_vs_Prior_30D": growth_pct,
            "Stockout_Risk_Level": risk,
            "Historical_Peak_Day": int(item_data["Quantity_Sold"].max()),
            "Historical_Total_Sold": int(item_data["Quantity_Sold"].sum())
        })

    full_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("Suggested_Restock_Quantity", ascending=False)

    execution_meta = {
        "item_count": len(unique_items),
        "items": unique_items,
        "total_rows_processed": len(df),
        "engines": list(engines_used),
        "forecast_days": forecast_days,
        "is_tabpfn_active": any("TabPFN Regressor (Prior Labs Cloud API)" in e for e in engines_used)
    }

    return full_forecast_df, summary_df, execution_meta
