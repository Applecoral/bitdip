# Required: pip install streamlit pandas numpy requests lightgbm pandas_ta scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pandas_ta as ta
from lightgbm import LGBMClassifier
import warnings

# Suppress warnings for a clean terminal output
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA & ENGINE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data():
    """Fetches 1m OHLC data from Kraken API."""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        resp = requests.get(url, timeout=10).json()
        if 'result' not in resp: return pd.DataFrame()
        ohlc = resp['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc, columns=['ts', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'cnt'])
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='s')
        df = df[['ts', 'open', 'high', 'low', 'close']].set_index('ts').apply(pd.to_numeric)
        return df
    except Exception:
        return pd.DataFrame()

def get_signal(df, timeframe):
    """Resamples data and runs LightGBM to predict the NEXT candle."""
    rule = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H'}.get(timeframe, '1T')
    df_tf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    if len(df_tf) < 30: 
        return "INIT...", 0.0
    
    # Feature Engineering
    df_tf['RSI'] = ta.rsi(df_tf['close'], length=14)
    # Target: 1 if next candle close > current candle close
    df_tf['target'] = (df_tf['close'].shift(-1) > df_tf['close']).astype(int)
    df_tf = df_tf.ffill().dropna()
    
    # ML Prediction
    try:
        X = df_tf[['RSI', 'close']].iloc[:-1]
        y = df_tf['target'].iloc[:-1]
        
        # Fast-training model (LGBM)
        model = LGBMClassifier(n_estimators=30, max_depth=3, verbosity=-1, importance_type='gain')
        model.fit(X, y)
        
        # Predict the NEXT outcome using the latest closed candle
        latest_data = df_tf[['RSI', 'close']].iloc[-1:]
        prob = model.predict_proba(latest_data)[0][1]
        
        side = "HIGHER" if prob > 0.5 else "LOWER"
        return side, prob
    except:
        return "ERROR", 0.5

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC_TERMINAL_V1", layout="centered")

# Custom CSS to force a Terminal Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');
    
    .stApp { background-color: #000000; }
    .terminal {
        font-family: 'Fira Code', monospace;
        color: #00FF41;
        background-color: #000000;
        padding: 30px;
        border: 1px solid #00FF41;
        border-radius: 5px;
        line-height: 1.5;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
    }
    .higher { color: #00FF41; font-weight: bold; }
    .lower { color: #FF3131; font-weight: bold; }
    .blink { animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
""", unsafe_allow_html=True)

def main():
    placeholder = st.empty()
    
    # Sidebar only for status
    st.sidebar.markdown("### `SYSTEM_STATUS`")
    st.sidebar.code("MODE: LIVE_STREAM\nENGINE: LIGHTGBM_v4.1\nFEED: KRAKEN_REST")

    while True:
        df = fetch_data()
        
        if not df.empty:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            price = f"{df['close'].iloc[-1]:,.2f}"
            
            # Generate predictions
            p1m, c1m = get_signal(df, '1m')
            p5m, c5m = get_signal(df, '5m')
            p15m, c15m = get_signal(df, '15m')
            p1h, c1h = get_signal(df, '1h')

            # Color coding helper
            def fmt(p): return f'<span class="{"higher" if p=="HIGHER" else "lower"}">{p}</span>'

            # Terminal Text Construction
            terminal_html = f"""
            <div class="terminal">
                <div>[SYSTEM] BTC_PULSE_MONITOR Initialized...</div>
                <div>[TIME  ] {now}</div>
                <div>[PRICE ] <span class="higher">${price}</span></div>
                <br>
                <div>------------------------------------------</div>
                <div>&gt; TF     | PREDICTION | CONFIDENCE</div>
                <div>------------------------------------------</div>
                <div>&gt; 1M     | {fmt(p1m)}     | {c1m if p1m=="HIGHER" else 1-c1m:.1%}</div>
                <div>&gt; 5M     | {fmt(p5m)}     | {c5m if p5m=="HIGHER" else 1-c5m:.1%}</div>
                <div>&gt; 15M    | {fmt(p15m)}     | {c15m if p15m=="HIGHER" else 1-c15m:.1%}</div>
                <div>&gt; 1H     | {fmt(p1h)}     | {c1h if p1h=="HIGHER" else 1-c1h:.1%}</div>
                <div>------------------------------------------</div>
                <br>
                <div class="blink">_ AWAITING_NEXT_TICK...</div>
            </div>
            """
            placeholder.markdown(terminal_html, unsafe_allow_html=True)
        else:
            placeholder.error("TERMINAL_ERROR: Data Link Severed. Retrying...")

        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
