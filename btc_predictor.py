import streamlit as st
import pandas as pd
import requests
import time
import pandas_ta as ta
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE (With Rate-Limit Handling)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_safe_data():
    """Fetches data with a retry buffer to prevent DATA_LINK_FAILURE."""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        # Use a longer timeout and check for status
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            ohlc = resp.json()['result']['XXBTZUSD']
            df = pd.DataFrame(ohlc, columns=['ts', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'cnt'])
            df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='s')
            df = df[['ts', 'open', 'high', 'low', 'close']].set_index('ts').apply(pd.to_numeric)
            return df.tail(720) # Kraken's max limit for 1m is 720
    except:
        pass
    return pd.DataFrame()

@st.cache_resource(ttl=600) # Cache models for 10 minutes to stay under 1GB RAM
def calculate_signal(df_resampled):
    if len(df_resampled) < 30:
        return "WAIT", 0.5
    
    try:
        # Minimal feature set to save memory
        df = df_resampled.copy()
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        
        X = df[['RSI', 'close']].iloc[:-1]
        y = df['target'].iloc[:-1]
        
        # Hyper-restricted model to prevent memory crashes
        model = LGBMClassifier(n_estimators=15, max_depth=2, verbosity=-1, num_leaves=4)
        model.fit(X, y)
        
        prob = model.predict_proba(df[['RSI', 'close']].iloc[-1:])[0][1]
        return ("HIGHER" if prob > 0.5 else "LOWER"), prob
    except:
        return "ERR", 0.5

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC_CORE", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #050505; }
    .term { font-family: 'Courier New', monospace; color: #00FF41; background: #000; padding: 15px; border: 1px solid #222; }
    .up { color: #00FF41; font-weight: bold; }
    .down { color: #FF3131; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.fragment(run_every=30)
def run_terminal():
    df_raw = fetch_safe_data()
    
    if not df_raw.empty:
        price = f"{df_raw['close'].iloc[-1]:,.2f}"
        tfs = {"1M": "1T", "5M": "5T", "15M": "15T", "1H": "1H", "1D": "1D"}
        lines = []
        
        for label, rule in tfs.items():
            # Resample one by one to keep memory usage linear, not exponential
            df_tf = df_raw.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
            side, prob = calculate_signal(df_tf)
            cls = "up" if side == "HIGHER" else "down"
            conf = prob if side == "HIGHER" else 1 - prob
            lines.append(f"[{label:>3}] {side:<6} | CONF: {conf:.1%}")

        st.markdown(f"""
        <div class="term">
            <div style="color:#888; border-bottom:1px solid #222; margin-bottom:10px;">CORE_TERMINAL_v2.7 // BTC-USD</div>
            <div>SYS_TIME: {time.strftime("%H:%M:%S")}</div>
            <div>BTC_PRICE: <span class="up">${price}</span></div>
            <br>
            {"<br>".join(lines)}
            <br>
            <div style="color:#444;">----------------------------------</div>
            <div>[OK]: SEQUENCE_COMPLETE. SLEEPING 30S...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="term">[!] DATA_LINK_FAILURE: RETRYING...</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run_terminal()
