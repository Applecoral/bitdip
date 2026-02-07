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

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING WITH RETRY LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def fetch_binance_klines(interval='1m', limit=1000, max_retries=3):
    """Fetch real 1-minute BTCUSDT candles from Binance with retry logic"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": min(limit, 1000)  # Binance max is 1000
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Empty or invalid response from Binance")
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to proper types with error handling
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                raise ValueError("All data was invalid after conversion")
            
            df = df.set_index('timestamp')
            
            # Store successful fetch
            st.session_state.last_data = df.copy()
            st.session_state.error_count = 0
            
            return df
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                st.session_state.error_count += 1
                return use_fallback_data(f"Network error: {str(e)}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                st.session_state.error_count += 1
                return use_fallback_data(f"Data error: {str(e)}")
    
    return use_fallback_data("Max retries exceeded")

def use_fallback_data(error_msg):
    """Use cached data if available, otherwise return empty DataFrame"""
    if st.session_state.last_data is not None and not st.session_state.last_data.empty:
        st.warning(f"⚠️ Using cached data due to: {error_msg}")
        return st.session_state.last_data.copy()
    else:
        st.error(f"❌ No data available: {error_msg}")
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
        
        # Validate resampled data
        if resampled.empty:
            return pd.DataFrame()
        
        # Ensure we have enough data points
        if len(resampled) < 5:
            return pd.DataFrame()
        
        return resampled
        
    except Exception as e:
        st.error(f"Resampling error for {timeframe}: {str(e)}")
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
            df['RSI_14'] = 50  # Neutral fallback
        
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
        
        # Fill any remaining NaN with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Final dropna for any still-missing values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        st.error(f"Feature engineering error: {str(e)}")
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
        
        # Define features to use
        base_features = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                        'BBM_20_2.0', 'returns']
        
        # Check which features are available
        available_features = [f for f in base_features if f in df_feat.columns]
        
        if len(available_features) < 3:
            return None, [], "Too few features available"
        
        X = df_feat[available_features]
        y = df_feat['target']
        
        # Validate data
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(0)
            y = y.fillna(0)
        
        # Check for variance
        if X.std().min() == 0:
            return None, [], "No variance in features"
        
        # Train model
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            verbose=-1,
            random_state=42,
            force_col_wise=True  # Suppress warnings
        )
        
        model.fit(X, y)
        
        return model, available_features, "success"
        
    except Exception as e:
        return None, [], f"Model training error: {str(e)}"

def predict_direction(df_resampled, timeframe):
    """Make prediction with fallback for errors"""
    try:
        # Get or train model
        model, feats, status = get_model_for_timeframe(df_resampled, timeframe)
        
        if model is None:
            return "Insufficient Data", 0.5, status
        
        if df_resampled.empty or len(df_resampled) < 10:
            return "Warming Up", 0.5, "Need more candles"
        
        # Get features for latest candle
        df_feat = engineer_features(df_resampled)
        
        if df_feat.empty:
            return "Feature Error", 0.5, "Could not compute indicators"
        
        # Extract latest features
        latest_features = df_feat[feats].iloc[-1:].values
        
        # Handle NaN in features
        if np.isnan(latest_features).any():
            latest_features = np.nan_to_num(latest_features, 0)
        
        # Make prediction
        prob = model.predict_proba(latest_features)[0]
        
        # Get probability of price going up
        up_prob = prob[1] if len(prob) > 1 else 0.5
        
        # Determine direction
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
    
    # Determine card color based on prediction
    if "Error" in prediction or "Insufficient" in prediction or "Warming" in prediction:
        color = "#ffcc00"  # Warning yellow
        display_text = prediction
        confidence_text = "Waiting for data..."
    else:
        color = "#00ffcc" if "HIGHER" in prediction else "#ff4b4b"
        display_text = prediction
        confidence_text = f"Confidence: {confidence:.1%}"
    
    # Get current price info
    if not df_tf.empty:
        current_price = df_tf['close'].iloc[-1]
        current_open = df_tf['open'].iloc[-1]
        price_change = ((current_price - current_open) / current_open) * 100
        price_color = "#00ffcc" if price_change >= 0 else "#ff4b4b"
        price_text = f"${current_price:,.2f} ({price_change:+.2f}%)"
    else:
        price_text = "Loading..."
        price_color = "#888"
    
    # Status indicator
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
    st.sidebar.caption("📊 Data Source: Binance Public API")
    st.sidebar.caption(f"🔄 Error Count: {st.session_state.error_count}")
    
    # Header
    st.title("🚀 BTC Pulse Predictor")
    st.caption("Real-time Bitcoin price direction predictions across multiple timeframes")
    
    # Fetch data with loading indicator
    with st.spinner("📡 Fetching live market data..."):
        df_1m = fetch_binance_klines('1m', limit=1000)
    
    # Check if we got data
    if df_1m.empty:
        st.error("❌ Unable to load market data. Please check your internet connection.")
        st.info("The app will attempt to reconnect automatically.")
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
            label="Bitcoin (BTCUSDT)",
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
            # Resample data
            df_tf = resample_to_timeframe(df_1m, tf)
            
            # Make prediction
            pred, conf, status = predict_direction(df_tf, tf)
            
            # Render card
            render_prediction_card(tf, df_tf, pred, conf, status)
    
    st.markdown("---")
    
    # Chart section
    st.subheader("📊 Price Chart (1 Hour Timeframe)")
    
    df_1h = resample_to_timeframe(df_1m, '1h')
    
    if not df_1h.empty and len(df_1h) > 0:
        try:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df_1h.index,
                open=df_1h['open'],
                high=df_1h['high'],
                low=df_1h['low'],
                close=df_1h['close'],
                name="BTCUSDT"
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
