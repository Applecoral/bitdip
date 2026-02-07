# Required: pip install streamlit pandas numpy requests lightgbm pandas_ta
import streamlit as st
import pandas as pd
import requests
import time
import pandas_ta as ta
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOGIC (Optimized)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data():
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        resp = requests.get(url, timeout=10).json()
        ohlc = resp['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc, columns=['ts', 'open', 'high', 'low', 'close', 'v', 'vol', 'cnt'])
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='s')
        df = df[['ts', 'open', 'high', 'low', 'close']].set_index('ts').apply(pd.to_numeric)
        return df
    except: return pd.DataFrame()

def get_signal(df, timeframe):
    rule = {'1m': '1T', '5m': '5T', '1h': '1H'}.get(timeframe, '1T')
    df_tf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(df_tf) < 20: return "---", 0.0
    
    # Feature Engineering
    df_tf['RSI'] = ta.rsi(df_tf['close'], length=14)
    df_tf['target'] = (df_tf['close'].shift(-1) > df_tf['close']).astype(int)
    df_tf = df_tf.ffill().dropna()
    
    # Quick Train & Predict
    X = df_tf[['RSI', 'close']].iloc[:-1]
    y = df_tf['target'].iloc[:-1]
    model = LGBMClassifier(n_estimators=20, verbosity=-1).fit(X, y)
    
    prob = model.predict_proba(df_tf[['RSI', 'close']].iloc[-1:]) [0][1]
    side = "HIGHER" if prob > 0.5 else "LOWER"
    return side, prob

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC Terminal", page_icon="📟")

# Terminal Styling
st.markdown("""
    <style>
    .reportview-container, .main { background: #000000; }
    .terminal-text { 
        font-family: 'Courier New', Courier, monospace; 
        color: #00FF00; 
        background-color: #000000; 
        padding: 20px; 
        line-height: 1.2;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    placeholder = st.empty()
    
    while True:
        df = fetch_data()
        if not df.empty:
            p_1m, c_1m = get_signal(df, '1m')
            p_5m, c_5m = get_signal(df, '5m')
            p_1h, c_1h = get_signal(df, '1h')
            
            now = time.strftime("%H:%M:%S")
            price = f"{df['close'].iloc[-1]:,.2f}"
            
            # Construct Terminal Output
            terminal_output = f"""
            <div class="terminal-text">
            [SYSTEM READY] - BTC_PULSE_v2.0.4<br>
            [TIMESTAMP] : {now}<br>
            [LIVE_PRICE]: ${price}<br>
            ---------------------------------------<br>
            > T_FRAME | PREDICTION | CONFIDENCE<br>
            > 1M      | {p_1m}      | {c_1m:.1%}<br>
            > 5M      | {p_5m}      | {c_5m:.1%}<br>
            > 1H      | {p_1h}      | {c_1h:.1%}<br>
            ---------------------------------------<br>
            [STATUS]   : Awaiting next candle...<br>
            [REFRESH]  : 30s
            </div>
            """
            placeholder.markdown(terminal_output, unsafe_allow_html=True)
        
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
