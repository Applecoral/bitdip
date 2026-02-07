# Required: pip install streamlit pandas numpy requests plotly lightgbm pandas_ta
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

# ── CONFIG & STYLING ─────────────────────────────────────────────────────────
st.set_page_config(page_title="BTC Pulse Predictor", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .prediction-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #00ffcc; background: #161b22; }
    .status-active { color: #00ffcc; font-weight: bold; }
    .warning-text { color: #ffcc00; }
    </style>
""", unsafe_allow_html=True)

# ── STATE ────────────────────────────────────────────────────────────────────
if 'history_log' not in st.session_state:
    st.session_state.history_log = deque(maxlen=100)
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = {tf: {"correct":0, "total":0} for tf in ["1m","5m","15m","1h","1d"]}

# ── DATA FETCHING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_coingecko_data(days=2):
    """Fetch recent BTC price history from CoinGecko (approx 5-min intervals)"""
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=minute&precision=full"
    try:
        resp = requests.get(url, timeout=10).json()
        if 'prices' not in resp or not resp['prices']:
            st.warning("CoinGecko returned empty data")
            return pd.DataFrame()
        
        df = pd.DataFrame(resp['prices'], columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Approximate OHLC (common for price-only feeds)
        df['open'] = df['close'].shift(1).fillna(df['close'])
        df['high'] = df['close'].rolling(window=5, min_periods=1).max()   # rough approx
        df['low']  = df['close'].rolling(window=5, min_periods=1).min()
        df['volume'] = 0  # not available here
        
        return df
    except Exception as e:
        st.error(f"CoinGecko fetch failed: {str(e)}. Retrying next refresh...")
        return pd.DataFrame()

def resample_to_timeframe(df, timeframe):
    """Resample approximate data to requested timeframe"""
    if df.empty:
        return pd.DataFrame()
    
    rule_map = {
        '1m': '1T', '5m': '5T', '15m': '15T', '1h': '60T', '1d': '1D'
    }
    rule = rule_map.get(timeframe, '5T')
    
    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(how='all')

# ── FEATURE ENGINEERING & PREDICTION ─────────────────────────────────────────
def engineer_features(df):
    if len(df) < 30:
        return pd.DataFrame()
    
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df['returns'] = df['close'].pct_change()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

@st.cache_resource(ttl=300)  # cache 5 min
def get_model_for_timeframe(df_resampled, timeframe):
    df_feat = engineer_features(df_resampled)
    if len(df_feat) < 15:
        return None, []
    
    features = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                'BBM_20_2.0', 'returns']
    
    available = [f for f in features if f in df_feat.columns]
    if len(available) < 4:
        return None, []
    
    X = df_feat[available]
    y = df_feat['target']
    
    model = LGBMClassifier(n_estimators=120, learning_rate=0.05, verbose=-1, random_state=42)
    model.fit(X, y)
    
    return model, available

def predict_direction(df_resampled, timeframe):
    model, feats = get_model_for_timeframe(df_resampled, timeframe)
    
    if model is None or len(df_resampled) == 0:
        return "Waiting for data", 0.0
    
    latest = df_resampled.tail(1)
    df_feat_latest = engineer_features(latest)
    
    if df_feat_latest.empty:
        return "Not enough recent data", 0.0
    
    try:
        latest_x = df_feat_latest[feats].iloc[-1:].values
        prob = model.predict_proba(latest_x)[0]
        up_prob = prob[1] if len(prob) > 1 else 0.5
        direction = "HIGHER" if up_prob > 0.5 else "LOWER"
        confidence = max(up_prob, 1 - up_prob)
        return direction, confidence
    except Exception:
        return "Prediction error", 0.0

# ── UI COMPONENTS ────────────────────────────────────────────────────────────
def render_prediction_card(tf, df_tf, prediction, confidence):
    if prediction in ["Waiting for data", "Not enough recent data"]:
        color = "#ffcc00"
        text = prediction
    else:
        color = "#00ffcc" if prediction == "HIGHER" else "#ff4b4b"
        text = prediction
    
    current_open = df_tf['open'].iloc[-1] if not df_tf.empty else 0
    time_remaining = "Calculating..."  # placeholder
    
    with st.container():
        st.markdown(f"""
            <div class="prediction-card" style="border-left-color: {color}">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1.3rem; font-weight: bold;">{tf}</span>
                    <span class="status-active">LIVE</span>
                </div>
                <h2 style="color: {color}; margin: 12px 0;">{text}</h2>
                <div style="font-size: 0.95rem; opacity: 0.85;">
                    Confidence: {confidence:.1%} | Open: ${current_open:,.2f} | {time_remaining}
                </div>
            </div>
        """, unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("⚙️ Settings")
    auto_refresh = st.sidebar.toggle("Auto-refresh (10s)", value=True)
    st.sidebar.caption("Using CoinGecko data (approx 5-min base) + resampling")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("BTC Pulse Predictor")
    
    with st.spinner("Loading fresh market data..."):
        df_base = fetch_coingecko_data(days=2)
        
        if df_base.empty:
            st.error("Couldn't load data from CoinGecko. Check internet or try later.")
            return
        
        current_price = df_base['close'].iloc[-1]
        st.markdown(f"<h1 style='font-size: 3.4rem; margin: 0;'>${current_price:,.2f}</h1>", unsafe_allow_html=True)
        
        tfs = ["1m", "5m", "15m", "1h", "1d"]
        prediction_cols = st.columns(len(tfs))
        
        for i, tf in enumerate(tfs):
            df_tf = resample_to_timeframe(df_base, tf)
            pred, conf = predict_direction(df_tf, tf)
            
            with prediction_cols[i]:
                render_prediction_card(tf, df_tf, pred, conf)
    
    # Chart (1h view)
    st.subheader("Recent Market (1h candles)")
    df_1h = resample_to_timeframe(df_base, '1h')
    if not df_1h.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_1h.index,
            open=df_1h['open'],
            high=df_1h['high'],
            low=df_1h['low'],
            close=df_1h['close'],
            name="BTC/USD"
        )])
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=20, b=10),
            height=420,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Last update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WAT • Data: CoinGecko API")

    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
