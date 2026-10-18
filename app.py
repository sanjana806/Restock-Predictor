"""
app.py - Restock-Predictor: Premium AI Inventory Intelligence Dashboard
==============================================================================
Project: Restock-Predictor (MVP for E-Commerce Inventory Forecasting)
Entry for: DEV.to Hacktoberfest Open-Source AI Challenge
Tech Stack: Streamlit + Plotly + TabPFN API + MongoDB Atlas + Pandas + Render
Aesthetic: Modern Obsidian & Electric Indigo Glassmorphism SaaS
==============================================================================
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone

# Interactive Data Visualization
import plotly.graph_objects as go
import plotly.express as px

# Internal Modular Services
import db
import predictor

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Restock-Predictor | TabPFN Tabular AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS: GLASSMORPHISM, NEON ACCENTS & SAAS REFINEMENTS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, kbd, samp, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top App Padding Reduction */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 95%;
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 16px;
    }

    /* Badges */
    .badge-bar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid transparent;
    }
    .badge-hack {
        background: rgba(225, 29, 72, 0.15);
        color: #FB7185;
        border-color: rgba(225, 29, 72, 0.3);
    }
    .badge-tabpfn {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border-color: rgba(99, 102, 241, 0.35);
    }
    .badge-mongo {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .badge-render {
        background: rgba(14, 165, 233, 0.15);
        color: #7DD3FC;
        border-color: rgba(14, 165, 233, 0.3);
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Modern Styled File Uploader */
    div[data-testid="stFileUploader"] {
        border-radius: 14px;
        background: rgba(17, 24, 39, 0.45);
        padding: 12px;
        border: 1px dashed rgba(99, 102, 241, 0.4);
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #818CF8;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 10px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0 16px;
        background-color: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #6366F1 !important;
    }

    /* Primary Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-1px);
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0F19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1F2937;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #374151;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION & API MANAGEMENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ System Architecture")
    
    # TabPFN Foundation Model Section
    st.markdown("#### 1. TabPFN Model Client")
    env_key = predictor.get_tabpfn_api_key()
    user_api_key = st.text_input(
        "Prior Labs API Key / Token",
        value=env_key if env_key else "",
        type="password",
        help="Obtain your TabPFN API token at https://platform.priorlabs.ai. If omitted, a high-fidelity local tabular regression model runs automatically."
    )
    effective_api_key = user_api_key.strip() if user_api_key.strip() else None

    if effective_api_key:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; color:#4ADE80; font-size:0.85rem; font-weight:600; margin-top:-6px; margin-bottom:12px;">
            <span>●</span> Prior Labs Cloud Engine Connected
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; color:#FBBF24; font-size:0.85rem; font-weight:600; margin-top:-6px; margin-bottom:12px;">
            <span>○</span> Local Tabular Regression Fallback
        </div>
        """, unsafe_allow_html=True)

    # MongoDB Atlas Status Section
    st.markdown("#### 2. MongoDB Atlas Cluster")
    with st.spinner("Pinging Atlas..."):
        mongo_info = db.test_connection()

    if mongo_info["connected"]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; color:#4ADE80; font-size:0.85rem; font-weight:600; margin-bottom:4px;">
            <span>●</span> Atlas Live: <code>{mongo_info['database']}</code>
        </div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:12px;">
            Collection: <code>{mongo_info['collection']}</code> ({mongo_info.get('doc_count', 0)} audits logged)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; color:#F87171; font-size:0.85rem; font-weight:600; margin-bottom:4px;">
            <span>○</span> Atlas: {mongo_info['status'].replace('_', ' ').title()}
        </div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:12px;">
            {mongo_info['message']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # One-Click Sample Dataset Demo Loader
    st.markdown("#### 🚀 Test Baseline")
    load_sample_trigger = st.button("Load Pre-Packaged Sample CSV", type="primary", use_container_width=True)
    st.caption("Loads 90 days of continuous multi-item sales across 3 Etsy/Shopify SKUs (`sample_data.csv`).")

    st.markdown("---")

    # Hacktoberfest Evaluation Summary
    with st.expander("🏆 Hacktoberfest & Render Info", expanded=False):
        st.markdown("""
        - **Challenge:** DEV.to Hacktoberfest Open-Source AI Challenge
        - **Prize Track:** $200 Best Use of Render
        - **AI Foundation Model:** TabPFN (`tabpfn-client`)
        - **Database:** MongoDB Atlas (`pymongo`)
        - **IaC Blueprint:** `render.yaml`
        """)

# -----------------------------------------------------------------------------
# 4. MAIN HERO SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Restock-Predictor</div>
    <div class="hero-sub">
        Next-generation inventory intelligence for Etsy & Shopify makers. Powered by the <b>TabPFN tabular foundation model</b> to deliver sub-second probabilistic 30-day demand forecasts and eliminate stockouts.
    </div>
    <div class="badge-bar">
        <span class="pill-badge badge-hack">🎃 DEV.to Hacktoberfest 2026</span>
        <span class="pill-badge badge-tabpfn">⚡ TabPFN Tabular In-Context AI</span>
        <span class="pill-badge badge-mongo">🍃 MongoDB Atlas Audit Engine</span>
        <span class="pill-badge badge-render">🚀 Render Native Deploy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DATA INGESTION & PARSING PIPELINE
# -----------------------------------------------------------------------------
col_upload, col_guidance = st.columns([1.6, 1.0])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Historical Store Sales CSV",
        type=["csv"],
        help="Upload sales transactions with columns: Date, Item_ID, Quantity_Sold."
    )

with col_guidance:
    st.markdown("""
    <div style="background:rgba(17,24,39,0.5); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:16px; font-size:0.85rem; color:#94A3B8;">
        <div style="font-weight:700; color:#F8FAFC; margin-bottom:6px;">Expected Schema Format:</div>
        <div>• <code>Date</code>: YYYY-MM-DD timestamp</div>
        <div>• <code>Item_ID</code>: Product SKU / Catalog Name</div>
        <div>• <code>Quantity_Sold</code>: Units purchased per day</div>
        <div style="margin-top:6px; color:#A5B4FC;">💡 Or click <b>'Load Pre-Packaged Sample CSV'</b> in sidebar!</div>
    </div>
    """, unsafe_allow_html=True)

raw_df = None
active_filename = None

if load_sample_trigger:
    sample_file_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    if os.path.exists(sample_file_path):
        raw_df = pd.read_csv(sample_file_path)
        active_filename = "sample_data.csv (Demo Dataset)"
        st.toast("Loaded sample dataset with 90 continuous days across 3 SKUs!", icon="⚡")
    else:
        st.error("sample_data.csv baseline not found on filesystem.")
elif uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        active_filename = uploaded_file.name
    except Exception as exc:
        st.error(f"Error parsing uploaded CSV: {exc}")

# -----------------------------------------------------------------------------
# 6. PROCESSING, AUDIT LOGGING & AI FORECASTING
# -----------------------------------------------------------------------------
if raw_df is not None:
    is_valid, validation_msg = predictor.validate_sales_csv(raw_df)
    
    if not is_valid:
        st.error(f"CSV Validation Failed: {validation_msg}")
        st.info("Please verify that your CSV contains the three headers: Date, Item_ID, Quantity_Sold.")
    else:
        # Normalize Data
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df["Quantity_Sold"] = pd.to_numeric(raw_df["Quantity_Sold"], errors="coerce").fillna(1)
        raw_df["Item_ID"] = raw_df["Item_ID"].astype(str)

        total_rows = len(raw_df)
        unique_items = list(raw_df["Item_ID"].unique())
        item_count = len(unique_items)
        date_start = raw_df["Date"].min().strftime("%b %d, %Y")
        date_end = raw_df["Date"].max().strftime("%b %d, %Y")
        total_units_sold = int(raw_df["Quantity_Sold"].sum())

        # Audit Log to MongoDB Atlas
        session_log_key = f"audit_{active_filename}_{total_rows}"
        if session_log_key not in st.session_state:
            audit_result = db.log_upload_summary(
                filename=active_filename,
                row_count=total_rows,
                item_count=item_count,
                items=unique_items,
                metadata={
                    "period": f"{date_start} to {date_end}",
                    "total_sales_volume": total_units_sold,
                    "engine": "TabPFN" if effective_api_key else "Local_Fallback"
                }
            )
            st.session_state[session_log_key] = audit_result

        # Run AI Inference Pipeline
        with st.spinner("Extracting tabular features & executing TabPFN in-context regression..."):
            try:
                forecast_df, summary_df, meta = predictor.run_inventory_forecasting(
                    raw_df,
                    forecast_days=30,
                    api_key=effective_api_key
                )
            except Exception as exc:
                st.error(f"Prediction engine failed: {exc}")
                st.stop()

        # Overall Metrics Aggregations
        total_predicted_demand = int(summary_df["Total_Predicted_Sales_30D"].sum())
        total_safety_buffer = int(summary_df["Recommended_Safety_Stock"].sum())
        total_suggested_restock = int(summary_df["Suggested_Restock_Quantity"].sum())
        
        # Calculate overall 30D historical comparison
        hist_cutoff = raw_df["Date"].max() - pd.Timedelta(days=30)
        recent_30d_sales = int(raw_df[raw_df["Date"] > hist_cutoff]["Quantity_Sold"].sum())
        overall_growth = round(((total_predicted_demand - recent_30d_sales) / max(1, recent_30d_sales)) * 100, 1)

        # ---------------------------------------------------------------------
        # 7. HIGH-LEVEL KPI METRIC CARDS (WITH COLORED DELTAS)
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                label="30-Day Predicted Demand",
                value=f"{total_predicted_demand:,} units",
                delta=f"{overall_growth:+.1f}% vs prior 30d",
                delta_color="normal"
            )

        with m2:
            buffer_pct = round((total_safety_buffer / max(1, total_predicted_demand)) * 100, 1)
            st.metric(
                label="90% Quantile Safety Buffer",
                value=f"+{total_safety_buffer:,} units",
                delta=f"{buffer_pct}% volatility buffer",
                delta_color="off"
            )

        with m3:
            st.metric(
                label="Total Restock Needed (PO)",
                value=f"{total_suggested_restock:,} units",
                delta="Demand + Buffer",
                delta_color="off"
            )

        with m4:
            engine_label = "TabPFN Cloud AI" if meta["is_tabpfn_active"] else "Local Tabular Fallback"
            st.metric(
                label="Active Catalog SKUs",
                value=f"{item_count} items",
                delta=engine_label,
                delta_color="normal"
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 8. SPACIOUS TAB NAVIGATION
        # ---------------------------------------------------------------------
        tab_forecast, tab_po, tab_insights, tab_mongo, tab_arch = st.tabs([
            "📈 Interactive Forecast & Buffers",
            "📋 Restock Purchase Orders",
            "📊 Demand Trends & Velocity",
            "🗄️ MongoDB Atlas Audit",
            "🧠 Why TabPFN vs LLMs"
        ])

        # =====================================================================
        # TAB 1: INTERACTIVE PLOTLY FORECAST & BUFFER VISUALIZATION
        # =====================================================================
        with tab_forecast:
            st.markdown("#### Dynamic Multi-Series Demand Forecasting with Confidence Ribbons")
            st.caption("Interactive time-series visualization showing historical sales, TabPFN predicted demand, and 90th-percentile safety buffer bands.")

            col_selector, col_view_type = st.columns([2, 1])
            with col_selector:
                item_options = ["All Catalog Items (Aggregated)"] + unique_items
                selected_sku = st.selectbox("Filter by Product SKU:", options=item_options)
            with col_view_type:
                chart_history_days = st.select_slider(
                    "Historical Context Window:",
                    options=[14, 30, 60, 90],
                    value=60,
                    format_func=lambda d: f"Last {d} Days"
                )

            # Filter data based on selection
            hist_min_date = raw_df["Date"].max() - pd.Timedelta(days=chart_history_days)
            filtered_raw = raw_df[raw_df["Date"] >= hist_min_date]

            if selected_sku == "All Catalog Items (Aggregated)":
                chart_hist = filtered_raw.groupby("Date")["Quantity_Sold"].sum().reset_index()
                chart_fc = forecast_df.groupby("Date")[["Predicted_Daily_Demand", "Safety_Buffer_90Pct"]].sum().reset_index()
                item_title = "All Store SKUs Combined"
            else:
                chart_hist = filtered_raw[filtered_raw["Item_ID"] == selected_sku].sort_values("Date")
                chart_fc = forecast_df[forecast_df["Item_ID"] == selected_sku].sort_values("Date")
                item_title = f"SKU: {selected_sku}"

            chart_fc["Date"] = pd.to_datetime(chart_fc["Date"])

            # Build Modern Plotly Figure
            fig = go.Figure()

            # 1. Historical Actuals (Sleek Slate Line)
            fig.add_trace(go.Scatter(
                x=chart_hist["Date"],
                y=chart_hist["Quantity_Sold"],
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(color="#94A3B8", width=2.5),
                marker=dict(size=4, color="#64748B"),
                hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Actual Sales:</b> %{y} units<extra></extra>"
            ))

            # 2. Safety Buffer Ceiling (Translucent upper band for fill)
            fig.add_trace(go.Scatter(
                x=chart_fc["Date"],
                y=chart_fc["Safety_Buffer_90Pct"],
                mode="lines",
                name="90% Safety Stock Ceiling",
                line=dict(color="rgba(245, 158, 11, 0.4)", width=1.5, dash="dash"),
                hoverinfo="skip"
            ))

            # 3. TabPFN Predicted Demand (Electric Violet/Indigo Glowing Line) with fill to safety buffer
            fig.add_trace(go.Scatter(
                x=chart_fc["Date"],
                y=chart_fc["Predicted_Daily_Demand"],
                mode="lines+markers",
                name="TabPFN Forecast (Expected)",
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.12)",  # Soft amber glow between mean and buffer
                line=dict(color="#818CF8", width=3.5),
                marker=dict(size=5, color="#6366F1"),
                hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Predicted Demand:</b> %{y} units<extra></extra>"
            ))

            # Connect historical last point to forecast first point for continuity
            bridge_x = [chart_hist["Date"].iloc[-1], chart_fc["Date"].iloc[0]]
            bridge_y = [chart_hist["Quantity_Sold"].iloc[-1], chart_fc["Predicted_Daily_Demand"].iloc[0]]
            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines",
                line=dict(color="#6366F1", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Vertical separator for Today / Forecast Origin
            last_date_val = chart_hist["Date"].max()
            fig.add_vline(
                x=last_date_val,
                line_width=1.5,
                line_dash="dot",
                line_color="#CBD5E1",
                annotation_text="Forecast Horizon Start",
                annotation_position="top left",
                annotation_font=dict(color="#CBD5E1", size=11)
            )

            # Modern Dark Theme Plotly Styling
            fig.update_layout(
                title=dict(
                    text=f"<b>Historical Sales vs. 30-Day TabPFN Forecast ({item_title})</b>",
                    font=dict(color="#F8FAFC", size=16)
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(17, 24, 39, 0.0)",
                plot_bgcolor="rgba(17, 24, 39, 0.4)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#CBD5E1", size=12)
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(
                    gridcolor="rgba(255, 255, 255, 0.06)",
                    showline=True,
                    linecolor="rgba(255, 255, 255, 0.1)",
                    tickformat="%b %d"
                ),
                yaxis=dict(
                    gridcolor="rgba(255, 255, 255, 0.06)",
                    showline=True,
                    linecolor="rgba(255, 255, 255, 0.1)",
                    title="Daily Units Sold / Needed"
                ),
                height=460
            )

            st.plotly_chart(fig, use_container_width=True)

        # =====================================================================
        # TAB 2: RESTOCK PURCHASE ORDER (PO) TABLE & COLOR GRADIENT STYLING
        # =====================================================================
        with tab_po:
            st.markdown("#### 📋 30-Day Purchase Order & Restock Plan")
            st.caption("Prioritized procurement recommendations calculated from TabPFN point forecasts + 90th percentile volatility buffers.")

            # Apply aesthetic gradient styling to dataframe
            st_styled_df = summary_df.copy()

            # Format numbers for clean presentation
            styled_view = (
                st_styled_df.style
                .background_gradient(subset=["Suggested_Restock_Quantity"], cmap="Purples")
                .background_gradient(subset=["Total_Predicted_Sales_30D"], cmap="Blues")
                .background_gradient(subset=["Recommended_Safety_Stock"], cmap="Oranges")
                .format({
                    "Suggested_Restock_Quantity": "{:,} units",
                    "Total_Predicted_Sales_30D": "{:,} units",
                    "Recommended_Safety_Stock": "+{:,} units",
                    "Avg_Daily_Demand": "{:.1f} / day",
                    "Growth_vs_Prior_30D": "{:+.1f}%",
                    "Historical_Peak_Day": "{:,} units",
                    "Historical_Total_Sold": "{:,} units"
                })
            )

            st.dataframe(styled_view, use_container_width=True, hide_index=True)

            # Purchase Order Export Options
            csv_export_buffer = io.StringIO()
            summary_df.to_csv(csv_export_buffer, index=False)
            
            c_down, c_tip = st.columns([1, 2])
            with c_down:
                st.download_button(
                    label="📥 Export Restock PO (CSV)",
                    data=csv_export_buffer.getvalue(),
                    file_name=f"restock_purchase_order_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            with c_tip:
                st.info("💡 **Procurement Rule:** Restock recommendations automatically include safety stock buffers to guarantee 90% in-stock availability during weekend demand surges.")

        # =====================================================================
        # TAB 3: DEMAND TRENDS, SKU VELOCITY & PEAKS
        # =====================================================================
        with tab_insights:
            st.markdown("#### 📊 SKU Velocity & Procurement Allocation")
            
            col_bar, col_pie = st.columns([1.5, 1.0])

            with col_bar:
                # Grouped Bar Chart: Expected Demand vs Safety Buffer Stock per SKU
                bar_fig = go.Figure()
                bar_fig.add_trace(go.Bar(
                    x=summary_df["Item_ID"],
                    y=summary_df["Total_Predicted_Sales_30D"],
                    name="Base Expected Sales",
                    marker_color="#6366F1"
                ))
                bar_fig.add_trace(go.Bar(
                    x=summary_df["Item_ID"],
                    y=summary_df["Recommended_Safety_Stock"],
                    name="Safety Buffer Stock",
                    marker_color="#F59E0B"
                ))
                bar_fig.update_layout(
                    barmode="stack",
                    title="<b>Restock Volume Composition by SKU</b>",
                    template="plotly_dark",
                    paper_bgcolor="rgba(17,24,39,0)",
                    plot_bgcolor="rgba(17,24,39,0.3)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=360,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Units")
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            with col_pie:
                # Donut Chart for Inventory Capital Allocation
                pie_fig = px.pie(
                    summary_df,
                    names="Item_ID",
                    values="Suggested_Restock_Quantity",
                    title="<b>Procurement Share by Item</b>",
                    hole=0.45,
                    color_discrete_sequence=["#818CF8", "#38BDF8", "#34D399", "#F472B6", "#FBBF24"]
                )
                pie_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(17,24,39,0)",
                    margin=dict(l=10, r=10, t=40, b=20),
                    height=360
                )
                st.plotly_chart(pie_fig, use_container_width=True)

        # =====================================================================
        # TAB 4: MONGODB ATLAS AUDIT LOGS & HEALTH
        # =====================================================================
        with tab_mongo:
            st.markdown("#### 🗄️ MongoDB Atlas Telemetry & Upload Audit Trails")
            st.caption("Every batch uploaded to Restock-Predictor is immutably audited in MongoDB Atlas with metadata, timestamps, and row counts.")

            recent_logs = db.get_recent_logs(limit=10)
            if recent_logs:
                formatted_logs = []
                for entry in recent_logs:
                    formatted_logs.append({
                        "Audit Timestamp": entry.get("timestamp"),
                        "Source File": entry.get("filename"),
                        "Processed Rows": entry.get("row_count"),
                        "Unique SKUs": entry.get("item_count"),
                        "Audit Storage": entry.get("status"),
                        "Record ID": entry.get("_id")
                    })
                st.table(pd.DataFrame(formatted_logs))
            else:
                st.write("No audit entries recorded in current session.")

        # =====================================================================
        # TAB 5: ARCHITECTURAL DEEP-DIVE: TABPFN VS LLMS
        # =====================================================================
        with tab_arch:
            st.markdown("#### 🧠 Why TabPFN Beats Standard LLMs for Tabular Forecasting")
            st.caption("Technical justification for the DEV.to Hacktoberfest Open-Source AI Challenge.")

            st.markdown("""
            | Comparison Dimension | Standard Generative LLMs (e.g. GPT-4 / Claude) | TabPFN Tabular Foundation Model (`tabpfn-client`) |
            | :--- | :--- | :--- |
            | **Numerical Representation** | Numbers are encoded as arbitrary Byte-Pair Encoding (BPE) text tokens. Prone to arithmetic failures and hallucinations. | Continuous numerical and categorical features are processed directly by an in-context transformer. |
            | **Inference Latency** | Sequential autoregressive text generation requires 5–15 seconds per forecast query. | **Sub-second single forward pass (< 500ms)** across the entire feature matrix. |
            | **Uncertainty Quantification** | Cannot output true statistical distribution quantiles; relies on descriptive adjectives. | **Native probabilistic quantiles** (`quantiles=[0.5, 0.9]`) enabling exact safety buffer sizing. |
            | **Zero-Shot Generalization** | LLMs struggle to respect strict calendar periodicity, moving averages, and lag correlations without extensive prompts. | TabPFN was trained on millions of synthetic tabular datasets to learn Bayesian regression natively. |
            | **Data Privacy & Cost** | Pushing 90 days of proprietary sales into prompt windows consumes thousands of expensive tokens. | Numerical vectors are passed directly without token bloat or chat prompt leakage. |
            """)

else:
    # Empty State Callout
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 48px 24px;">
        <div style="font-size: 2.8rem; margin-bottom: 12px;">📦</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin-bottom: 8px;">
            Ready to Forecast Your Next 30 Days of Inventory
        </div>
        <div style="font-size: 0.95rem; color: #94A3B8; max-width: 600px; margin: 0 auto 24px auto;">
            Upload your historical store sales CSV above, or click the <b>'Load Pre-Packaged Sample CSV'</b> button in the sidebar to test with a 90-day multi-item dataset immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748B;">
    <div>Restock-Predictor MVP • DEV.to Hacktoberfest Open-Source AI Challenge</div>
    <div>Streamlit • Plotly • TabPFN • MongoDB Atlas • Render</div>
</div>
""", unsafe_allow_html=True)
