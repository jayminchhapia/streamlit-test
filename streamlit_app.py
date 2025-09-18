import os
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="NSE/BSE Predictor - Working Version", layout="wide")
INDIA_TZ = pytz.timezone("Asia/Kolkata")

@st.cache_resource(show_spinner=False)
def get_finbert():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return None

def is_market_open_now():
    now = datetime.now(INDIA_TZ)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now.weekday() < 5 and (now >= open_t) and (now <= close_t)

def resolve_ticker(user_query: str):
    q = user_query.strip().upper()
    candidates = [q] if (q.endswith(".NS") or q.endswith(".BO")) else [f"{q}.NS", f"{q}.BO"]
    
    st.write("🔍 Testing candidates:", candidates)
    
    for ticker in candidates:
        try:
            df = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if df is not None and len(df.dropna()) > 10:
                st.write(f"✅ Resolved to: {ticker}")
                return ticker
            else:
                st.write(f"❌ {ticker}: No sufficient data")
        except Exception as e:
            st.write(f"❌ {ticker}: Error - {str(e)}")
    
    st.error("Could not resolve any valid ticker")
    return None

def fetch_price_data(ticker: str, start_date: date, end_date: date):
    st.write(f"📈 Fetching price data for {ticker}")
    st.write(f"Period: {start_date} to {end_date}")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        
        if df is None or len(df) == 0:
            st.error("No price data returned")
            return pd.DataFrame()
        
        # Ensure we have Close column
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        
        # Keep essential columns and ensure no negative prices
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df[df["Close"] > 0]  # Remove any invalid price data
        
        st.write(f"✅ Price data: {len(df)} rows")
        min_price = df['Close'].min()
        max_price = df['Close'].max()
        st.write(f"Price range: ₹{min_price:.2f} to ₹{max_price:.2f}")
        st.dataframe(df.head(5))
        
        return df
        
    except Exception as e:
        st.error(f"Price fetch failed: {str(e)}")
        return pd.DataFrame()

def fetch_news_data(ticker: str, limit: int = 10):
    st.write(f"📰 Fetching news for {ticker}")
    
    try:
        t = yf.Ticker(ticker)
        news_raw = getattr(t, 'news', None) or []
        
        if not news_raw:
            st.write("⚠️ No news available from yfinance")
            return []
        
        valid_items = []
        
        for item in news_raw[:limit]:
            title = item.get("title", "").strip()
            publisher = item.get("publisher", "").strip()
            link = item.get("link", "").strip()
            timestamp = item.get("providerPublishTime")
            
            if not title or not timestamp or timestamp <= 0:
                continue
                
            try:
                pub_date = datetime.fromtimestamp(timestamp, tz=INDIA_TZ)
                valid_items.append({
                    "title": title,
                    "publisher": publisher or "Unknown",
                    "link": link or "No link",
                    "pubDate": pub_date
                })
            except (ValueError, OSError):
                continue
        
        st.write(f"✅ Found {len(valid_items)} valid headlines")
        
        if valid_items:
            st.subheader("📰 Latest Headlines")
            for i, item in enumerate(valid_items[:5], 1):
                st.write(f"**{i}.** {item['title']}")
                st.write(f"   📍 Source: {item['publisher']}")
        
        return valid_items
        
    except Exception as e:
        st.write(f"⚠️ News fetch error: {str(e)}")
        return []

def analyze_sentiment(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate", "title", "sentiment", "score"])
    
    pipe = get_finbert()
    if pipe is None:
        st.write("⚠️ FinBERT not available")
        return pd.DataFrame(columns=["pubDate", "title", "sentiment", "score"])
    
    try:
        titles = [h["title"] for h in headlines]
        results = pipe(titles, truncation=True)
        
        sentiment_data = []
        for headline, result in zip(headlines, results):
            sentiment_data.append({
                "pubDate": headline["pubDate"],
                "title": headline["title"],
                "sentiment": result["label"],
                "score": float(result["score"])
            })
        
        df = pd.DataFrame(sentiment_data)
        sentiment_counts = df["sentiment"].value_counts()
        st.write("📊 Sentiment Distribution:", dict(sentiment_counts))
        
        return df
        
    except Exception as e:
        st.write(f"⚠️ Sentiment analysis failed: {str(e)}")
        return pd.DataFrame(columns=["pubDate", "title", "sentiment", "score"])

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_technical_indicators(price_df):
    """Compute technical indicators with proper bounds checking"""
    df = price_df.copy()
    
    # Price-based features (percentage changes are bounded)
    df['ret_1'] = df['Close'].pct_change().clip(-0.2, 0.2)  # Limit to ±20%
    df['ret_5'] = df['Close'].pct_change(5).clip(-0.5, 0.5)  # Limit to ±50%
    
    # Moving averages (use ratios to normalize)
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['ma_ratio'] = (df['sma_5'] / df['sma_20']).clip(0.8, 1.2)  # Reasonable range
    
    # RSI (already bounded 0-100)
    df['rsi'] = compute_rsi(df['Close']).clip(0, 100)
    
    # Volume ratio (normalize by median)
    median_vol = df['Volume'].rolling(20).median()
    df['vol_ratio'] = (df['Volume'] / median_vol).clip(0.1, 5.0)
    
    # Volatility (using rolling std of returns)
    df['volatility'] = df['ret_1'].rolling(10).std().clip(0, 0.1)
    
    return df

def create_model_dataset(price_df, sentiment_df, horizon_days):
    """Create dataset with proper target variable and feature validation"""
    st.write(f"🏗️ Creating dataset for {horizon_days}-day prediction")
    
    # Compute technical indicators
    df = compute_technical_indicators(price_df)
    
    # Add date column
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df.index.values if 'Date' not in df.columns else df['Date'])
    
    # Add sentiment features if available
    if sentiment_df is not None and len(sentiment_df) > 0:
        # Aggregate sentiment by day
        sent_daily = sentiment_df.copy()
        sent_daily['Date'] = pd.to_datetime(sent_daily['pubDate']).dt.date
        sent_daily['Date'] = pd.to_datetime(sent_daily['Date'])
        
        # Convert sentiment to numeric
        sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
        sent_daily['sent_numeric'] = sent_daily['sentiment'].map(sentiment_map) * sent_daily['score']
        
        # Aggregate by date
        sent_agg = sent_daily.groupby('Date').agg({
            'sent_numeric': ['mean', 'count']
        }).round(4)
        sent_agg.columns = ['sent_score', 'sent_count']
        sent_agg = sent_agg.reset_index()
        
        # Merge with price data
        df = pd.merge(df, sent_agg, on='Date', how='left')
        df['sent_score'] = df['sent_score'].fillna(0).clip(-1, 1)
        df['sent_count'] = df['sent_count'].fillna(0).clip(0, 10)
    else:
        df['sent_score'] = 0.0
        df['sent_count'] = 0.0
    
    # Create target variable (percentage change instead of absolute price)
    df = df.sort_values('Date')
    future_prices = df['Close'].shift(-horizon_days)
    df['target_pct_change'] = ((future_prices - df['Close']) / df['Close']).clip(-0.3, 0.3)  # Limit to ±30%
    
    # Select features (all normalized/bounded)
    feature_columns = [
        'ret_1', 'ret_5', 'ma_ratio', 'rsi', 'vol_ratio', 'volatility',
        'sent_score', 'sent_count'
    ]
    
    # Create final dataset
    df_clean = df[['Date', 'Close'] + feature_columns + ['target_pct_change']].dropna()
    
    if len(df_clean) < 50:
        st.error(f"Insufficient data: {len(df_clean)} samples. Need at least 50.")
        return None, None, None, None, None
    
    X = df_clean[feature_columns].values
    y = df_clean['target_pct_change'].values  # Predicting percentage change
    dates = df_clean['Date'].values
    current_price = df_clean['Close'].iloc[-1]
    
    # Validation: Check for reasonable values
    st.write(f"✅ Dataset created: {len(X)} samples")
    y_min = y.min() * 100
    y_max = y.max() * 100
    st.write(f"Target range: {y_min:.2f}% to {y_max:.2f}%")
    st.write(f"Current price: ₹{current_price:.2f}")
    
    return X, y, dates, feature_columns, current_price

def train_prediction_model(price_df, sentiment_df, horizon_days):
    """Train model with proper validation and bounds checking"""
    
    # Create dataset
    result = create_model_dataset(price_df, sentiment_df, horizon_days)
    if result[0] is None:
        return None
    
    X, y, dates, feature_columns, current_price = result
    
    st.write("🤖 Training prediction model")
    
    # Split data (80/20)
    split_point = int(0.8 * len(X))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest for stability
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Predict next value
    X_latest = scaler.transform(X[-1:])
    predicted_pct_change = model.predict(X_latest)[0]
    
    # Convert percentage change back to price
    predicted_price = current_price * (1 + predicted_pct_change)
    
    # Sanity check: limit prediction to reasonable bounds
    max_change = 0.15  # Max 15% change
    if abs(predicted_pct_change) > max_change:
        predicted_pct_change = np.sign(predicted_pct_change) * max_change
        predicted_price = current_price * (1 + predicted_pct_change)
        st.warning(f"Prediction capped at ±15% for safety")
    
    st.write("✅ Model training complete")
    mae_pct = test_mae * 100
    st.write(f"Test MAE: {mae_pct:.2f}%")
    st.write(f"Test R²: {test_r2:.3f}")
    change_pct = predicted_pct_change * 100
    st.write(f"Predicted change: {change_pct:+.2f}%")
    
    return {
        "predicted_price": predicted_price,
        "current_price": current_price,
        "predicted_change": predicted_pct_change,
        "mae": test_mae,
        "r2": test_r2,
        "n_samples": len(X)
    }

def generate_trading_signal(current_price, predicted_price, predicted_change, horizon):
    """Generate trading signal with realistic thresholds"""
    
    price_change_pct = predicted_change * 100
    
    # Define realistic thresholds
    if horizon == "INTRADAY":
        buy_threshold = 0.5   # 0.5%
        sell_threshold = -0.5 # -0.5%
        stop_pct = 1.5
    elif horizon == "SHORT":
        buy_threshold = 1.0   # 1.0%
        sell_threshold = -1.0 # -1.0%
        stop_pct = 2.5
    else:  # LONG
        buy_threshold = 2.0   # 2.0%
        sell_threshold = -2.0 # -2.0%
        stop_pct = 4.0
    
    # Generate signal
    if price_change_pct > buy_threshold:
        action = "BUY"
        stop_loss = current_price * (1 - stop_pct / 100)
    elif price_change_pct < sell_threshold:
        action = "SELL"
        stop_loss = current_price * (1 + stop_pct / 100)
    else:
        action = "HOLD"
        stop_loss = None
    
    direction = "UP" if price_change_pct > 0.2 else ("DOWN" if price_change_pct < -0.2 else "NEUTRAL")
    
    return action, current_price, stop_loss, predicted_price, price_change_pct, direction

def calculate_position_size(entry_price, stop_price, max_capital=15000):
    if entry_price <= 0:
        return 0, 0
    
    if stop_price and stop_price > 0:
        risk_per_share = abs(entry_price - stop_price)
        max_risk = max_capital * 0.01  # 1% risk
        quantity = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
    else:
        quantity = int(max_capital * 0.05 / entry_price)  # 5% of capital
    
    quantity = max(1, min(quantity, int(max_capital / entry_price)))
    total_cost = quantity * entry_price
    
    return quantity, total_cost

# Streamlit UI
st.title("🚀 NSE/BSE Stock Predictor - Working Version")
st.write(f"**Market Status:** {'🟢 OPEN' if is_market_open_now() else '🔴 CLOSED'}")

user_input = st.text_input("Enter stock symbol:", value="RELIANCE")
horizon = st.selectbox("Trading Horizon:", ["INTRADAY", "SHORT", "LONG"])
lookback = st.slider("Historical data (days):", min_value=180, max_value=730, value=365)

if st.button("🔍 Analyze Stock", type="primary"):
    ticker = resolve_ticker(user_input)
    if not ticker:
        st.stop()
    
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback)
    price_data = fetch_price_data(ticker, start_date, end_date)
    
    if price_data.empty:
        st.error("No price data available")
        st.stop()
    
    news_headlines = fetch_news_data(ticker, limit=10)
    sentiment_data = analyze_sentiment(news_headlines)
    
    horizon_days = {"INTRADAY": 1, "SHORT": 5, "LONG": 20}[horizon]
    model_result = train_prediction_model(price_data, sentiment_data, horizon_days)
    
    if model_result is None:
        st.error("Could not train model with available data")
        st.stop()
    
    action, entry, stop_loss, target, price_change, direction = generate_trading_signal(
        model_result["current_price"], 
        model_result["predicted_price"], 
        model_result["predicted_change"], 
        horizon
    )
    
    quantity, investment = calculate_position_size(entry, stop_loss)
    
    # Display results
    st.success("✅ Analysis Complete!")
    
    st.subheader("📋 Trading Recommendation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stock", ticker)
        st.metric("Action", action)
        st.metric("Current Price", f"₹{entry:.2f}")
        st.metric("Predicted Price", f"₹{target:.2f}")
    
    with col2:
        st.metric("Expected Move", f"{price_change:+.2f}%")
        st.metric("Direction", direction)
        if stop_loss:
            st.metric("Stop Loss", f"₹{stop_loss:.2f}")
        st.metric("Investment", f"₹{investment:,.0f}")
    
    # Model performance
    with st.expander("🤖 Model Performance"):
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        mae_display = model_result['mae'] * 100
        perf_col1.metric("Mean Absolute Error", f"{mae_display:.2f}%")
        perf_col2.metric("R² Score", f"{model_result['r2']:.3f}")
        perf_col3.metric("Training Samples", model_result['n_samples'])
    
    st.subheader("📈 Price History")
    chart_data = price_data["Close"].reset_index()
    chart_data.columns = ['Date', 'Close']
    st.line_chart(chart_data.set_index('Date'))
    
    # Summary
    st.subheader("📝 Trade Summary")
    st.write(f"**Stock:** {ticker}")
    st.write(f"**Date:** {datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    st.write(f"**Term:** {horizon}")
    st.write(f"**Market Open:** {'YES' if is_market_open_now() else 'NO'}")
    st.write(f"**Action:** {action}")
    st.write(f"**Price:** ₹{entry:.2f}")
    if stop_loss:
        st.write(f"**Stop Loss:** ₹{stop_loss:.2f}")
    st.write(f"**Predicted Price:** ₹{target:.2f}")
    st.write(f"**Expected Move:** {price_change:+.2f}% ({direction})")
    st.write(f"**Quantity:** {quantity} shares")
    st.write(f"**Investment:** ₹{investment:,.0f}")
    
    st.info("⚠️ This is for educational purposes only. Not financial advice. Always consult professionals before trading.")
