# Required: pip install streamlit pandas numpy requests plotly lightgbm pandas_ta scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import datetime
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
import pandas_ta as ta
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="BTC Pulse Predictor Pro", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .prediction-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #00ffcc; background: #161b22; min-height: 180px; }
    .status-active { color: #00ffcc; font-weight: bold; font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
if 'last_data' not in st.session_state:
    st.session_state.last_data = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'current_source' not in st.session_state:
    st.session_state.current_source = None

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_from_coinbase():
    try:
        # Coinbase returns 300 candles max. 
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {"granularity": 60} 
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').set_index('timestamp')
        return df
    except:
        return pd.DataFrame()

def fetch_from_kraken(limit=720):
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ohlc_data = data['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
        df = df.apply(pd.to_numeric)
        return df.tail(limit)
    except:
        return pd.DataFrame()

def fetch_from_coingecko():
    try:
        # Pull 30 days of data to satisfy the 1H and 1D timeframe needs
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "30", "interval": "hourly"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        df = prices.set_index('timestamp')
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] # Approximation for Gecko
        df['low'] = df['close']
        df['volume'] = 0
        return df.dropna()
    except:
        return pd.DataFrame()

def fetch_crypto_data_multi_source():
    sources = [
        ("Kraken", fetch_from_kraken),
        ("Coinbase", fetch_from_coinbase),
        ("CoinGecko", fetch_from_coingecko)
    ]
    
    for source_name, fetch_func in sources:
        try:
            df = fetch_func()
            if not df.empty and len(df) > 20:
                st.session_state.current_source = source_name
                st.session_state.last_data = df
                return df
        except:
            continue
    
    return st.session_state.last_data if st.session_state.last_data is not None else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def resample_to_timeframe(df, timeframe):
    if df.empty: return pd.DataFrame()
    rule_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '1d': '1D'}
    rule = rule_map.get(timeframe, '1T')
    return df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

def engineer_features(df):
    if len(df) < 15: return pd.DataFrame()
    df = df.copy()
    try:
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        df['returns'] = df['close'].pct_change()
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        return df.fillna(method='bfill').ffill()
    except:
        return pd.DataFrame()

def predict_direction(df_tf, tf):
    df_feat = engineer_features(df_tf)
    if df_feat.empty or len(df_feat) < 20:
        return "Insufficient Data", 0.5, "waiting"
    
    try:
        # Use simple features that are likely to exist
        feature_cols = ['RSI_14', 'returns'] 
        # Add MACD column dynamically as name varies by lib version
        macd_col = [c for c in df_feat.columns if 'MACD_' in c][:1]
        feature_cols += macd_col
        
        X = df_feat[feature_cols].iloc[:-1]
        y = df_feat['target'].iloc[:-1]
        
        model = LGBMClassifier(n_estimators=50, max_depth=3, verbosity=-1)
        model.fit(X, y)
        
        latest_row = df_feat[feature_cols].iloc[-1:]
        prob = model.predict_proba(latest_row)[0][1]
        
        direction = "HIGHER ↑" if prob > 0.5 else "LOWER ↓"
        return direction, max(prob, 1-prob), "success"
    except Exception as e:
        return "Error", 0.5, str(e)

# ══════════════════════════════════════════════════════════════════════════════
# UI & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_prediction_card(tf, df_tf, prediction, confidence, status):
    is_error = status != "success"
    color = "#ffcc00" if is_error else ("#00ffcc" if "HIGHER" in prediction else "#ff4b4b")
    
    current_price = f"${df_tf['close'].iloc[-1]:,.2f}" if not df_tf.empty else "---"
    conf_text = f"Confidence: {confidence:.1%}" if not is_error else "Gathering market pulse..."

    st.markdown(f"""
        <div class="prediction-card" style="border-left-color: {color}">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 1.2rem; font-weight: bold;">{tf.upper()}</span>
                <span class="status-active">● LIVE</span>
            </div>
            <h2 style="color: {color}; margin: 10px 0;">{prediction}</h2>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                {conf_text}<br/>
                <span style="font-weight: bold; color: white;">Price: {current_price}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.sidebar.title("⚙️ Settings")
    auto_refresh = st.sidebar.toggle("Auto-refresh (30s)", value=True)
    
    st.title("🚀 BTC Pulse Predictor")
    
    with st.spinner("Syncing with Global Exchanges..."):
        df_raw = fetch_crypto_data_multi_source()

    if df_raw.empty:
        st.error("No data connection. Check internet or API status.")
        return

    # Metrics
    c1, c2, c3 = st.columns(3)
    curr = df_raw['close'].iloc[-1]
    change = ((curr - df_raw['close'].iloc[-2]) / df_raw['close'].iloc[-2]) * 100
    c1.metric("BTC/USD", f"${curr:,.2f}", f"{change:+.2f}%")
    c2.metric("24h High", f"${df_raw['high'].max():,.2f}")
    c3.metric("Source", st.session_state.current_source)

    # Predictions
    st.write("### Market Direction Forecast")
    tfs = ["1m", "5m", "15m", "1h", "1d"]
    cols = st.columns(len(tfs))
    for i, tf in enumerate(tfs):
        with cols[i]:
            df_tf = resample_to_timeframe(df_raw, tf)
            pred, conf, status = predict_direction(df_tf, tf)
            render_prediction_card(tf, df_tf, pred, conf, status)

    # Chart
    fig = go.Figure(data=[go.Candlestick(x=df_raw.index, open=df_raw['open'], 
                    high=df_raw['high'], low=df_raw['low'], close=df_raw['close'])])
    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=0, b=0, l=0, r=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
