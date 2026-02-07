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
st.set_page_config(page_title="BTC Pulse Predictor", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .prediction-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #00ffcc; background: #161b22; }
    .status-active { color: #00ffcc; font-weight: bold; }
    .warning-text { color: #ffcc00; }
    .error-text { color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
if 'history_log' not in st.session_state:
    st.session_state.history_log = deque(maxlen=100)
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = {tf: {"correct": 0, "total": 0} for tf in ["1m", "5m", "15m", "1h", "1d"]}
if 'last_data' not in st.session_state:
    st.session_state.last_data = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'current_source' not in st.session_state:
    st.session_state.current_source = None

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SOURCE DATA FETCHING (FIXES GEO-BLOCKING)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_from_coinbase(limit=1000):
    """Fetch from Coinbase Pro API - no geo-blocking"""
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        
        # Coinbase uses granularity in seconds
        params = {
            "granularity": 60  # 1 minute candles
        }
        
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        
        # Coinbase format: [time, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
        
        # Limit to requested amount
        df = df.tail(limit)
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_from_kraken(limit=1000):
    """Fetch from Kraken API - reliable alternative"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            "pair": "XBTUSD",
            "interval": 1,  # 1 minute
            "since": int(time.time()) - (limit * 60)  # Get last N minutes
        }
        
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if 'result' not in data or 'XXBTZUSD' not in data['result']:
            return pd.DataFrame()
        
        # Kraken format: [time, open, high, low, close, vwap, volume, count]
        ohlc_data = data['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.dropna()
        df = df.set_index('timestamp')
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_from_coingecko(limit=1000):
    """Fetch from CoinGecko API - free and reliable"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        
        # CoinGecko uses days parameter
        days = max(1, limit // 1440 + 1)  # Convert minutes to days
        
        params = {
            "vs_currency": "usd",
            "days": min(days, 90),  # Max 90 days for free tier
            "interval": "minute" if days <= 1 else "hourly"
        }
        
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if 'prices' not in data or len(data['prices']) == 0:
            return pd.DataFrame()
        
        # CoinGecko only provides price data, we need to construct OHLC
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        
        # Resample to 1-minute and create OHLC
        prices = prices.set_index('timestamp')
        df = prices.resample('1T').agg({
            'close': ['first', 'max', 'min', 'last']
        })
        
        df.columns = ['open', 'high', 'low', 'close']
        df['volume'] = 0  # CoinGecko free tier doesn't provide volume in this endpoint
        
        df = df.dropna()
        df = df.tail(limit)
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_crypto_data_multi_source(limit=1000):
    """Try multiple sources to avoid geo-blocking issues"""
    sources = [
        ("Coinbase", fetch_from_coinbase),
        ("Kraken", fetch_from_kraken),
        ("CoinGecko", fetch_from_coingecko)
    ]
    
    for source_name, fetch_func in sources:
        try:
            df = fetch_func(limit)
            
            if not df.empty and len(df) >= 10:
                st.session_state.current_source = source_name
                st.session_state.last_data = df.copy()
                st.session_state.error_count = 0
                return df
                
        except Exception as e:
            continue
    
    # All sources failed, use cached data
    if st.session_state.last_data is not None and not st.session_state.last_data.empty:
        st.warning("⚠️ Using cached data - all live sources unavailable")
        return st.session_state.last_data.copy()
    
    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# DATA RESAMPLING
# ══════════════════════════════════════════════════════════════════════════════
def resample_to_timeframe(df, timeframe):
    """Convert 1m data to requested timeframe with validation"""
    if df.empty:
        return pd.DataFrame()
    
    rule_map = {
        '1m': '1T', 
        '5m': '5T', 
        '15m': '15T', 
        '1h': '1H', 
        '1d': '1D'
    }
    
    rule = rule_map.get(timeframe, '5T')
    
    try:
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if resampled.empty or len(resampled) < 5:
            return pd.DataFrame()
        
        return resampled
        
    except Exception as e:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df, min_periods=30):
    """Create technical indicators with robust error handling"""
    if df.empty or len(df) < min_periods:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        
        # RSI
        try:
            df.ta.rsi(length=14, append=True)
        except:
            df['RSI_14'] = 50
        
        # MACD
        try:
            df.ta.macd(append=True)
        except:
            df['MACD_12_26_9'] = 0
            df['MACDs_12_26_9'] = 0
            df['MACDh_12_26_9'] = 0
        
        # Bollinger Bands
        try:
            df.ta.bbands(length=20, std=2, append=True)
        except:
            df['BBM_20_2.0'] = df['close']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Target (for training)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Fill NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def get_model_for_timeframe(df_resampled, timeframe, min_samples=20):
    """Train model with comprehensive error handling"""
    try:
        df_feat = engineer_features(df_resampled)
        
        if df_feat.empty or len(df_feat) < min_samples:
            return None, [], "Insufficient data"
        
        base_features = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                        'BBM_20_2.0', 'returns']
        
        available_features = [f for f in base_features if f in df_feat.columns]
        
        if len(available_features) < 3:
            return None, [], "Too few features available"
        
        X = df_feat[available_features]
        y = df_feat['target']
        
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(0)
            y = y.fillna(0)
        
        if X.std().min() == 0:
            return None, [], "No variance in features"
        
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            verbose=-1,
            random_state=42,
            force_col_wise=True
        )
        
        model.fit(X, y)
        
        return model, available_features, "success"
        
    except Exception as e:
        return None, [], f"Model training error: {str(e)}"

def predict_direction(df_resampled, timeframe):
    """Make prediction with fallback for errors"""
    try:
        model, feats, status = get_model_for_timeframe(df_resampled, timeframe)
        
        if model is None:
            return "Insufficient Data", 0.5, status
        
        if df_resampled.empty or len(df_resampled) < 10:
            return "Warming Up", 0.5, "Need more candles"
        
        df_feat = engineer_features(df_resampled)
        
        if df_feat.empty:
            return "Feature Error", 0.5, "Could not compute indicators"
        
        latest_features = df_feat[feats].iloc[-1:].values
        
        if np.isnan(latest_features).any():
            latest_features = np.nan_to_num(latest_features, 0)
        
        prob = model.predict_proba(latest_features)[0]
        up_prob = prob[1] if len(prob) > 1 else 0.5
        
        direction = "HIGHER ↑" if up_prob > 0.5 else "LOWER ↓"
        confidence = max(up_prob, 1 - up_prob)
        
        return direction, confidence, "success"
        
    except Exception as e:
        return "Prediction Error", 0.5, str(e)

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def render_prediction_card(tf, df_tf, prediction, confidence, status):
    """Render prediction card with status indicators"""
    
    if "Error" in prediction or "Insufficient" in prediction or "Warming" in prediction:
        color = "#ffcc00"
        display_text = prediction
        confidence_text = "Waiting for data..."
    else:
        color = "#00ffcc" if "HIGHER" in prediction else "#ff4b4b"
        display_text = prediction
        confidence_text = f"Confidence: {confidence:.1%}"
    
    if not df_tf.empty:
        current_price = df_tf['close'].iloc[-1]
        current_open = df_tf['open'].iloc[-1]
        price_change = ((current_price - current_open) / current_open) * 100
        price_color = "#00ffcc" if price_change >= 0 else "#ff4b4b"
        price_text = f"${current_price:,.2f} ({price_change:+.2f}%)"
    else:
        price_text = "Loading..."
        price_color = "#888"
    
    status_icon = "🟢" if status == "success" else "🟡"
    
    st.markdown(f"""
        <div class="prediction-card" style="border-left-color: {color}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.4rem; font-weight: bold;">{tf.upper()}</span>
                <span class="status-active">{status_icon} LIVE</span>
            </div>
            <h2 style="color: {color}; margin: 15px 0 10px 0; font-size: 1.8rem;">{display_text}</h2>
            <div style="font-size: 0.95rem; opacity: 0.85; line-height: 1.6;">
                {confidence_text}<br/>
                <span style="color: {price_color};">{price_text}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    auto_refresh = st.sidebar.toggle("Auto-refresh (15s)", value=False)
    show_debug = st.sidebar.toggle("Show debug info", value=False)
    
    st.sidebar.markdown("---")
    
    if st.session_state.current_source:
        st.sidebar.success(f"📊 Data Source: {st.session_state.current_source}")
    else:
        st.sidebar.info("📊 Data Source: Connecting...")
    
    st.sidebar.caption(f"🔄 Error Count: {st.session_state.error_count}")
    st.sidebar.caption("💡 Using geo-blocking resistant APIs")
    
    # Header
    st.title("🚀 BTC Pulse Predictor")
    st.caption("Real-time Bitcoin price direction predictions across multiple timeframes")
    
    # Fetch data with loading indicator
    with st.spinner("📡 Fetching live market data..."):
        df_1m = fetch_crypto_data_multi_source(limit=1000)
    
    # Check if we got data
    if df_1m.empty:
        st.error("❌ Unable to load market data from all sources.")
        st.info("Trying: Coinbase → Kraken → CoinGecko")
        st.info("The app will retry automatically. If this persists, check requirements.txt includes 'requests'")
        
        if auto_refresh:
            time.sleep(15)
            st.rerun()
        return
    
    # Display current price
    current_price = df_1m['close'].iloc[-1]
    prev_price = df_1m['close'].iloc[-2] if len(df_1m) > 1 else current_price
    price_change_pct = ((current_price - prev_price) / prev_price) * 100
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.metric(
            label="Bitcoin (BTC/USD)",
            value=f"${current_price:,.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )
    with col2:
        st.metric(label="24h High", value=f"${df_1m['high'].tail(1440).max():,.2f}")
    with col3:
        st.metric(label="24h Low", value=f"${df_1m['low'].tail(1440).min():,.2f}")
    
    st.markdown("---")
    
    # Multi-timeframe predictions
    st.subheader("📈 Multi-Timeframe Predictions")
    
    timeframes = ["1m", "5m", "15m", "1h", "1d"]
    cols = st.columns(len(timeframes))
    
    for i, tf in enumerate(timeframes):
        with cols[i]:
            df_tf = resample_to_timeframe(df_1m, tf)
            pred, conf, status = predict_direction(df_tf, tf)
            render_prediction_card(tf, df_tf, pred, conf, status)
    
    st.markdown("---")
    
    # Chart section
    st.subheader("📊 Price Chart (1 Hour Timeframe)")
    
    df_1h = resample_to_timeframe(df_1m, '1h')
    
    if not df_1h.empty and len(df_1h) > 0:
        try:
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
                margin=dict(l=20, r=20, t=40, b=20),
                height=450,
                xaxis_rangeslider_visible=False,
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart rendering error: {str(e)}")
    else:
        st.warning("Not enough data to display chart")
    
    # Debug info
    if show_debug:
        st.markdown("---")
        st.subheader("🔧 Debug Information")
        st.write(f"**Data points available:** {len(df_1m)}")
        st.write(f"**Latest timestamp:** {df_1m.index[-1]}")
        st.write(f"**Data source:** {st.session_state.current_source}")
        st.write(f"**Cached data available:** {'Yes' if st.session_state.last_data is not None else 'No'}")
        with st.expander("View raw data sample"):
            st.dataframe(df_1m.tail(10))
    
    # Footer
    st.markdown("---")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.caption(f"⏰ Last updated: {current_time} | 🔄 Auto-refresh: {'ON' if auto_refresh else 'OFF'}")
    st.caption("⚠️ Disclaimer: This is for educational purposes only. Not financial advice.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(15)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.info("Please refresh the page to restart the application.")
        if st.button("🔄 Refresh Now"):
            st.rerun()
