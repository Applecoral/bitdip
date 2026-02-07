import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pandas_ta as ta
from lightgbm import LGBMClassifier
import warnings

# Optimization: Suppress lightgbm and pandas warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA & ML ENGINE (Optimized for Streamlit Cloud)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data():
    """Fetches OHLC data with a row limit to prevent memory spikes."""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        resp = requests.get(url, timeout=10).json()
        ohlc = resp['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc, columns=['ts', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'cnt'])
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='s')
        df = df[['ts', 'open', 'high', 'low', 'close']].set_index('ts').apply(pd.to_numeric)
        # Keep only the last 720 rows (enough for 1D timeframe resample)
        return df.tail(1000)
    except Exception:
        return pd.DataFrame()

@st.cache_resource(ttl=300) # Cache the model for 5 mins to save RAM
def train_and_predict(df_tf):
    """Handles the ML logic in a cached block to prevent memory overflow."""
    if len(df_tf) < 20: 
        return "INIT...", 0.5
    
    df_tf = df_tf.copy()
    df_tf['RSI'] = ta.rsi(df_tf['close'], length=14)
    df_tf['target'] = (df_tf['close'].shift(-1) > df_tf['close']).astype(int)
    df_tf = df_tf.ffill().dropna()
    
    try:
        X = df_tf[['RSI', 'close']].iloc[:-1]
        y = df_tf['target'].iloc[:-1]
        # Hyper-light model setup
        model = LGBMClassifier(n_estimators=20, max_depth=2, verbosity=-1)
        model.fit(X, y)
        
        prob = model.predict_proba(df_tf[['RSI', 'close']].iloc[-1:])[0][1]
        side = "HIGHER" if prob > 0.5 else "LOWER"
        return side, prob
    except:
        return "ERROR", 0.5

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC_TERMINAL", layout="centered")

# Terminal Styling
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .terminal-container {
        font-family: 'Courier New', Courier, monospace;
        color: #00FF41;
        background-color: #000000;
        padding: 20px;
        border: 1px solid #00FF41;
        border-radius: 5px;
    }
    .higher { color: #00FF41; font-weight: bold; }
    .lower { color: #FF3131; font-weight: bold; }
    .header { color: #FFFFFF; border-bottom: 1px solid #333; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# Using a fragment to refresh specific parts without a full app crash
@st.fragment(run_every=30)
def terminal_display():
    df = fetch_data()
    
    if not df.empty:
        now = time.strftime("%H:%M:%S")
        price = f"{df['close'].iloc[-1]:,.2f}"
        
        tfs = {"1m": "1T", "5m": "5T", "15m": "15T", "1h": "1H", "1d": "1D"}
        results = []
        
        for name, rule in tfs.items():
            df_tf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            side, prob = train_and_predict(df_tf)
            
            color = "higher" if side == "HIGHER" else "lower"
            conf = prob if side == "HIGHER" else 1 - prob
            results.append(f"[{name.upper():>3}] PREDICTION: <span class='{color}'>{side:<6}</span> | CONF: {conf:.1%}")

        output = f"""
        <div class="terminal-container">
            <div class="header">BTC_PULSE_TERMINAL v2.1 // SOURCE: KRAKEN</div>
            <div>TIMESTAMP: {now} | STATUS: CONNECTED</div>
            <div>LIVE_BTC : <span class="higher">${price}</span></div>
            <br>
            {"<br>".join(results)}
            <br>
            <div>--------------------------------------------------</div>
            <div>[SYSTEM]: Analyzing market pulse...</div>
        </div>
        """
        st.markdown(output, unsafe_allow_html=True)
    else:
        st.error("SYSTEM_ERROR: Data Link Severed.")

def main():
    terminal_display()

if __name__ == "__main__":
    main()
