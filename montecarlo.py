import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Monte Carlo Risk/Reward", layout="wide")
st.title("🎲 Monte Carlo & Risk Reward Analyzer")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Simulation Settings")

ticker = st.sidebar.text_input("Stock Ticker (Yahoo Finance)", value="BBRI.JK")
start_date = st.sidebar.date_input("Start Date for Analysis", value=pd.to_datetime("2019-01-01"))
sim_years = st.sidebar.slider("Years to Simulate", 0.5, 5.0, 1.0)
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1500)

# --- MAIN LOGIC ---

if ticker:
    with st.spinner(f'Analyzing {ticker}...'):
        try:
            # 1. GET DATA
            df = yf.download(ticker, start=start_date)
            
            # Handle MultiIndex if necessary
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    data = df['Close'][ticker]
                except KeyError:
                    data = df['Close'].iloc[:, 0]
            else:
                data = df['Close']
            
            data = data.squeeze().dropna()
            last_price = float(data.iloc[-1])
            
            st.success(f"Current Price: {last_price:,.2f}")
            
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()

    # 2. CALCULATE MONTE CARLO PARAMETERS
    log_returns = np.log(1 + data.pct_change()).dropna()
    
    u = log_returns.mean()
    if isinstance(u, (pd.Series, np.ndarray)): u = u.item()
    var = log_returns.var()
    if isinstance(var, (pd.Series, np.ndarray)): var = var.item()
    stdev = log_returns.std()
    if isinstance(stdev, (pd.Series, np.ndarray)): stdev = stdev.item()

    drift = u - (0.5 * var)
    
    days_to_sim = int(sim_years * 252)
    
    # 3. RUN SIMULATION
    Z = np.random.normal(0, 1, (days_to_sim, n_sims))
    daily_returns = np.exp(drift + stdev * Z)
    
    price_paths = np.zeros((days_to_sim, n_sims))
    price_paths[0] = last_price
    
    for t in range(1, days_to_sim):
        price_paths[t] = price_paths[t-1] * daily_returns[t]

    # 4. VISUALIZATION
    fig = go.Figure()

    # Scatter lines (Blue Haze)
    for i in range(min(n_sims, 100)):
        fig.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', 
                                 line=dict(width=1, color='rgba(0, 0, 255, 0.1)'), showlegend=False))

    # Average Path (Orange)
    fig.add_trace(go.Scatter(y=price_paths.mean(axis=1), mode='lines', name='Average Path',
                             line=dict(color='orange', width=3)))

    # Optimistic (Green)
    p95 = np.percentile(price_paths, 95, axis=1)
    fig.add_trace(go.Scatter(y=p95, mode='lines', name='Optimistic (95%)',
                             line=dict(color='green', width=2, dash='dash')))
    
    # Pessimistic (Red)
    p05 = np.percentile(price_paths, 5, axis=1)
    fig.add_trace(go.Scatter(y=p05, mode='lines', name='Pessimistic (5%)',
                             line=dict(color='red', width=2, dash='dash')))

    fig.update_layout(title=f"Projected Path for {ticker}", template="plotly_white", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # 5. RISK REWARD CALCULATION
    final_prices = price_paths[-1]
    
    upside_price = np.percentile(final_prices, 95)
    downside_price = np.percentile(final_prices, 5)
    
    # Calculate Distances
    reward_dist = upside_price - last_price
    risk_dist = last_price - downside_price
    
    # Avoid division by zero
    if risk_dist <= 0:
        rr_ratio = 0
    else:
        rr_ratio = reward_dist / risk_dist

    # --- DISPLAY METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Expected Price", f"{final_prices.mean():,.0f}")
    c2.metric("Upside (95%)", f"{upside_price:,.0f}", delta=f"+{reward_dist:,.0f}")
    c3.metric("Downside Risk (5%)", f"{downside_price:,.0f}", delta=f"-{risk_dist:,.0f}", delta_color="inverse")
    
    # Custom Color Logic for Score
    if rr_ratio >= 2.0:
        score_color = "normal" # Green in Streamlit
        label = "⭐ Great Setup"
    elif rr_ratio >= 1.0:
        score_color = "off" # Grey/Yellowish
        label = "⚖️ Neutral"
    else:
        score_color = "inverse" # Red
        label = "⚠️ Bad Risk"

    c4.metric("Risk/Reward Ratio", f"{rr_ratio:.2f}", delta=label, delta_color=score_color)