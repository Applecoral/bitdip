import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pandas_ta as ta
from lightgbm import LGBMClassifier
import warnings

# Optimization: Silence LightGBM logs to keep the terminal clean
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA & ENGINE (Memory-Safe)
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
        # RAM SAFETY: Only keep enough for analysis
        return df.tail(1000)
    except:
        return pd.DataFrame()

@st.cache_resource(ttl=300) # Only re-train models every 5 minutes to stay under 1GB RAM
def get_ml_signal(df_tf):
    if len(df_tf) < 25: return "INIT...", 0.5
    
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
# THE TERMINAL VIEW
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC_TERMINAL", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    .terminal-box {
        font-family: 'Courier New', Courier, monospace;
        color: #00FF41;
        background-color: #000000;
        padding: 20px;
        border: 1px solid #00FF41;
        border-radius: 4px;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }
    .higher { color: #00FF41; font-weight: bold; }
    .lower { color: #FF3131; font-weight: bold; }
    .header { color: #FFFFFF; border-bottom: 1px solid #333; margin-bottom: 12px; }
    </style>
""", unsafe_allow_html=True)

@st.fragment(run_every=30)
def live_terminal():
    df = fetch_data()
    
    if not df.empty:
        now = time.strftime("%H:%M:%S")
        price = f"{df['close'].iloc[-1]:,.2f}"
        
        tfs = {"1M": "1T", "5M": "5T", "15M": "15T", "1H": "1H", "1D": "1D"}
        results = []
        
        for label, rule in tfs.items():
            df_tf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            side, prob = get_ml_signal(df_tf)
            
            color = "higher" if side == "HIGHER" else "lower"
            conf = prob if side == "HIGHER" else 1 - prob
            results.append(f"[{label:>3}] PREDICTION: <span class='{color}'>{side:<6}</span> | CONF: {conf:.1%}")

        terminal_html = f"""
        <div class="terminal-box">
            <div class="header">BTC_PULSE_TERMINAL_v2.5 // LIVE_FEED</div>
            <div>UTC_TIME: {now} | STATUS: CONNECTED</div>
            <div>PRICE_USD: <span class="higher">${price}</span></div>
            <br>
            {"<br>".join(results)}
            <br>
            <div>------------------------------------------</div>
            <div>[SYSTEM]: Awaiting next candle...</div>
        </div>
        """
        st.markdown(terminal_html, unsafe_allow_html=True)
    else:
        st.error("DATA_LINK_FAILURE: Retrying connection...")

if __name__ == "__main__":
    live_terminal()
