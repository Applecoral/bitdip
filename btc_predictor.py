# Required: pip install streamlit pandas numpy requests lightgbm pandas_ta scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pandas_ta as ta
from lightgbm import LGBMClassifier
import warnings

# Suppress warnings for clean terminal output
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE (Intact from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data():
    """Fetches raw 1m OHLC data."""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        resp = requests.get(url, timeout=10).json()
        ohlc = resp['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc, columns=['ts', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'cnt'])
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='s')
        df = df[['ts', 'open', 'high', 'low', 'close']].set_index('ts').apply(pd.to_numeric)
        return df
    except:
        return pd.DataFrame()

def get_signal(df, timeframe):
    """Calculates ML signal for specified timeframe."""
    rule = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '1d': '1D'}.get(timeframe, '1T')
    df_tf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    if len(df_tf) < 20: return "FETCHING", 0.0
    
    # Feature Engineering (Intact)
    df_tf['RSI'] = ta.rsi(df_tf['close'], length=14)
    df_tf['target'] = (df_tf['close'].shift(-1) > df_tf['close']).astype(int)
    df_tf = df_tf.ffill().dropna()
    
    try:
        X = df_tf[['RSI', 'close']].iloc[:-1]
        y = df_tf['target'].iloc[:-1]
        model = LGBMClassifier(n_estimators=30, max_depth=3, verbosity=-1)
        model.fit(X, y)
        
        prob = model.predict_proba(df_tf[['RSI', 'close']].iloc[-1:])[0][1]
        side = "HIGHER" if prob > 0.5 else "LOWER"
        return side, prob
    except:
        return "ERROR", 0.5

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI (Text-Only)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC_TERMINAL", layout="centered")

# CSS to simulate a terminal environment
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .terminal-container {
        font-family: 'Courier New', Courier, monospace;
        color: #00FF41;
        background-color: #000000;
        padding: 20px;
        line-height: 1.4;
    }
    .higher { color: #00FF41; font-weight: bold; }
    .lower { color: #FF3131; font-weight: bold; }
    .header { color: #FFFFFF; border-bottom: 1px solid #00FF41; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

def main():
    # Empty placeholder for the terminal feed
    terminal_ui = st.empty()
    
    while True:
        df = fetch_data()
        
        if not df.empty:
            now = time.strftime("%H:%M:%S")
            price = f"{df['close'].iloc[-1]:,.2f}"
            
            # Gather Predictions
            tfs = ["1m", "5m", "15m", "1h", "1d"]
            results = []
            for tf in tfs:
                side, prob = get_signal(df, tf)
                color_class = "higher" if side == "HIGHER" else "lower"
                conf = prob if side == "HIGHER" else 1 - prob
                results.append(f"[{tf.upper():>3}] PREDICTION: <span class='{color_class}'>{side:<6}</span> | CONF: {conf:.1%}")

            # Building the Terminal String
            output = f"""
            <div class="terminal-container">
                <div class="header">BTC_PULSE_TERMINAL v2.0 // STATUS: CONNECTED</div>
                <div>TIMESTAMP: {now}</div>
                <div>LIVE_BTC : <span class="higher">${price}</span></div>
                <br>
                {"<br>".join(results)}
                <br>
                <div>--------------------------------------------------</div>
                <div>[SYSTEM]: Awaiting next candle close...</div>
            </div>
            """
            terminal_ui.markdown(output, unsafe_allow_html=True)
        else:
            terminal_ui.markdown('<div class="terminal-container">ERROR: DATA_FEED_OFFLINE</div>', unsafe_allow_html=True)

        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
