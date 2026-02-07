# Required libraries: pip install streamlit pandas numpy requests plotly lightgbm pandas_ta
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

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="BTC Pulse Predictor", layout="wide", page_icon="📈")

# Professional Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .prediction-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #00ffcc; background: #161b22; }
    .status-active { color: #00ffcc; font-weight: bold; }
    .error-text { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'history_log' not in st.session_state:
    st.session_state.history_log = deque(maxlen=100)
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'accuracy_stats' not in st.session_state:
    st.session_state.accuracy_stats = {"1m": {"correct":0, "total":0}, "5m": {"correct":0, "total":0},
                                       "15m": {"correct":0, "total":0}, "1h": {"correct":0, "total":0},
                                       "1d": {"correct":0, "total":0}}

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def fetch_coingecko_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=5).json()
        return response['bitcoin']['usd'], response['bitcoin']['usd_24h_change']
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def fetch_ohlc(days=2):  # increased to 2 days → more data points
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=10).json()
        df = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')  # helps with potential resampling later
        return df
    except Exception as e:
        st.warning(f"OHLC fetch failed: {str(e)}")
        return pd.DataFrame()

def fetch_coinglass_metrics(api_key):
    if not api_key:
        return {"oi": 15400000000, "funding": 0.01, "ls_ratio": 1.05}
    
    headers = {'CG-API-KEY': api_key}
    base = "https://open-api-v4.coinglass.com/api/futures/"
    try:
        oi_resp = requests.get(base + "openInterest?symbol=BTC", headers=headers, timeout=5).json()
        oi = oi_resp.get('data', [{}])[0].get('openInterest', 0)
        
        # Add funding rate fetch (example - adjust endpoint if needed)
        fund_resp = requests.get(base + "fundingRate?symbol=BTC", headers=headers, timeout=5).json()
        funding = fund_resp.get('data', [{}])[0].get('fundingRate', 0.01)
        
        return {"oi": oi, "funding": funding, "ls_ratio": 1.02}  # ls_ratio placeholder
    except:
        return {"oi": 0, "funding": 0.01, "ls_ratio": 1.0}

# --- ML ENGINE ---
def engineer_features(df):
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df['returns'] = df['close'].pct_change()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

@st.cache_resource(ttl=600)  # cache model 10 min
def get_trained_model(df_feat):
    if df_feat.empty or len(df_feat) < 20:
        return None
    
    features = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'BBM_20_2.0', 'returns']
    
    # Safe column check
    available_features = [f for f in features if f in df_feat.columns]
    if len(available_features) < 4:  # arbitrary minimum
        return None
    
    X = df_feat[available_features]
    y = df_feat['target']
    
    model = LGBMClassifier(n_estimators=150, learning_rate=0.04, verbose=-1, random_state=42)
    model.fit(X, y)
    return model, available_features

def train_and_predict(df_feat, timeframe):
    model_tuple = get_trained_model(df_feat)
    if model_tuple is None:
        return "Not enough data", 0.0
    
    model, feats = model_tuple
    
    if len(df_feat) == 0:
        return "No data", 0.0
    
    latest_x = df_feat[feats].iloc[-1:].values
    prob = model.predict_proba(latest_x)[0]
    up_prob = prob[1] if len(prob) > 1 else 0.5
    prediction = "HIGHER" if up_prob > 0.5 else "LOWER"
    confidence = max(up_prob, 1 - up_prob)
    
    return prediction, confidence

# --- UI COMPONENTS ---
def render_prediction_card(tf, open_price, current_price, prediction, confidence):
    if prediction in ["Not enough data", "No data"]:
        color = "#ffcc00"
        text = prediction
    else:
        color = "#00ffcc" if prediction == "HIGHER" else "#ff4b4b"
        text = prediction
    
    with st.container():
        st.markdown(f"""
            <div class="prediction-card" style="border-left-color: {color}">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1.2rem; font-weight: bold;">{tf} Timeframe</span>
                    <span class="status-active">LIVE</span>
                </div>
                <h2 style="color: {color}; margin: 10px 0;">{text}</h2>
                <div style="font-size: 0.9rem; opacity: 0.8;">
                    Confidence: {confidence:.2%} | Open: ${open_price:,.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    st.sidebar.title("⚙️ Controls")
    coinglass_key = st.sidebar.text_input("CoinGlass API Key (optional)", type="password")
    auto_refresh = st.sidebar.toggle("Auto-Refresh every 10s", value=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    btc_price, change = fetch_coingecko_price()
    
    with col1:
        st.title("Bitcoin Prediction Dashboard")
        if btc_price:
            st.markdown(f"<h1 style='font-size: 3rem;'>${btc_price:,.2f} <span style='font-size: 1rem; color: {'#00ffcc' if (change or 0) > 0 else '#ff4b4b'}'>{change:+.2f}%</span></h1>", unsafe_allow_html=True)
        else:
            st.error("Could not fetch BTC price — check internet")
    
    with st.spinner("Analyzing market patterns..."):
        ohlc_data = fetch_ohlc(days=2)  # more data
        derivatives = fetch_coinglass_metrics(coinglass_key)
        
        if ohlc_data.empty:
            st.error("No OHLC data available from CoinGecko. Try again later.")
        else:
            df_feat = engineer_features(ohlc_data)
            
            tfs = ["1m", "5m", "15m", "1h", "1d"]
            cols = st.columns(len(tfs))
            
            for i, tf in enumerate(tfs):
                with cols[i]:
                    pred, conf = train_and_predict(df_feat, tf)
                    # Placeholder open price (last open)
                    open_price = ohlc_data['open'].iloc[-1] if not ohlc_data.empty else 0
                    render_prediction_card(tf, open_price, btc_price, pred, conf)
    
    st.subheader("Market Visualization")
    if not ohlc_data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name="BTC/USD"
        )])
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Derivatives Insights (CoinGlass)")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Open Interest", f"${derivatives['oi']/1e9:.2f}B", delta=None)
    m_col2.metric("Funding Rate", f"{derivatives['funding']*100:.4f}%", delta=None)
    m_col3.metric("Long/Short Ratio", f"{derivatives['ls_ratio']:.2f}", delta=None)
    
    st.divider()
    st.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
