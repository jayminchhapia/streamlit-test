import os
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta
from transformers import pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="NSE/BSE Predictor - Fixed", layout="wide")
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
        
        # Keep essential columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        
        st.write(f"✅ Price data: {len(df)} rows")
        st.dataframe(df.head(5))
        
        return df
        
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        return pd.DataFrame()

def fetch_news_data(ticker: str, limit: int = 15):
    st.write(f"📰 Fetching news for {ticker}")
    
    try:
        t = yf.Ticker(ticker)
        news_raw = getattr(t, 'news', None) or []
        
        if not news_raw:
            st.write("⚠️ No news available from yfinance")
            return []
        
        valid_items = []
        
        for item in news_raw[:limit]:
            # Extract fields safely
            title = item.get("title", "").strip()
            publisher = item.get("publisher", "").strip()
            link = item.get("link", "").strip()
            timestamp = item.get("providerPublishTime")
            
            # Skip if no title or invalid timestamp
            if not title or not timestamp or timestamp <= 0:
                continue
                
            # Convert timestamp
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
        
        # Display headlines clearly
        if valid_items:
            st.subheader("📰 Latest Headlines")
            for i, item in enumerate(valid_items[:10], 1):
                st.write(f"**{i}.** {item['title']}")
                st.write(f"   📍 Source: {item['publisher']}")
                if item['link'] != "No link":
                    st.write(f"   🔗 [Read more]({item['link']})")
                st.write("---")
        else:
            st.write("No headlines with valid timestamps found")
        
        return valid_items
        
    except Exception as e:
        st.write(f"⚠️ News fetch error: {e}")
        return []

def analyze_sentiment(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate", "title", "sentiment", "score", "publisher"])
    
    pipe = get_finbert()
    if pipe is None:
        st.write("⚠️ FinBERT not available, skipping sentiment analysis")
        return pd.DataFrame(columns=["pubDate", "title", "sentiment", "score", "publisher"])
    
    st.write(f"🧠 Analyzing sentiment for {len(headlines)} headlines")
    
    # Extract titles for sentiment analysis
    titles = [h["title"] for h in headlines]
    
    try:
        # Run sentiment analysis
        results = pipe(titles, truncation=True)
        
        # Combine results
        sentiment_data = []
        for headline, result in zip(headlines, results):
            sentiment_data.append({
                "pubDate": headline["pubDate"],
                "title": headline["title"],
                "sentiment": result["label"],
                "score": float(result["score"]),
                "publisher": headline["publisher"]
            })
        
        df = pd.DataFrame(sentiment_data)
        
        # Show sentiment distribution
        sentiment_counts = df["sentiment"].value_counts()
        st.write("📊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            st.write(f"   {sentiment}: {count}")
        
        return df
        
    except Exception as e:
        st.write(f"⚠️ Sentiment analysis failed: {e}")
        return pd.DataFrame(columns=["pubDate", "title", "sentiment", "score", "publisher"])

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def build_technical_features(price_data: pd.DataFrame):
    st.write("🔧 Building technical features")
    
    df = price_data.copy()
    
    # Technical indicators
    df["ret_1"] = df["Close"].pct_change()
    df["ret_5"] = df["Close"].pct_change(5)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["VIX"] = 15.0  # Default VIX value
    
    # Remove NaN rows
    df = df.dropna()
    
    # Convert index to Date column properly
    df = df.reset_index()
    df = df.rename(columns={"Date": "Date"})
    
    # Ensure Date column is proper datetime
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Date"] = pd.to_datetime(df["Date"])
    
    st.write(f"✅ Technical features: {len(df)} rows")
    st.dataframe(df[["Date", "Close", "ret_1", "ma_ratio", "rsi_14"]].head(5))
    
    return df

def aggregate_daily_sentiment(sentiment_df: pd.DataFrame):
    if sentiment_df is None or len(sentiment_df) == 0:
        st.write("⚠️ No sentiment data to aggregate")
        return pd.DataFrame(columns=["Date", "sent_mean", "sent_count", "sent_pos", "sent_neg"])
    
    st.write("📊 Aggregating sentiment by day")
    
    df = sentiment_df.copy()
    
    # Convert to date only
    df["Date"] = pd.to_datetime(df["pubDate"]).dt.date
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Map sentiment to numeric
    sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
    df["sentiment_score"] = df["sentiment"].str.lower().map(sentiment_map).fillna(0) * df["score"]
    
    # Aggregate by date
    daily_sentiment = df.groupby("Date").agg({
        "sentiment_score": ["mean", "count", lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
    }).round(4)
    
    # Flatten column names
    daily_sentiment.columns = ["sent_mean", "sent_count", "sent_pos", "sent_neg"]
    daily_sentiment = daily_sentiment.reset_index()
    
    st.write(f"✅ Daily sentiment: {len(daily_sentiment)} days")
    st.dataframe(daily_sentiment.head(5))
    
    return daily_sentiment

def create_final_dataset(tech_df: pd.DataFrame, sent_df: pd.DataFrame, horizon_days: int):
    st.write(f"🏗️ Creating dataset for {horizon_days}-day prediction")
    
    # Start with technical data
    final_df = tech_df.copy()
    
    # Add sentiment data if available
    if sent_df is not None and len(sent_df) > 0:
        final_df = pd.merge(final_df, sent_df, on="Date", how="left")
        st.write("✅ Merged with sentiment data")
    else:
        # Add empty sentiment columns
        for col in ["sent_mean", "sent_count", "sent_pos", "sent_neg"]:
            final_df[col] = 0.0
        st.write("⚠️ No sentiment data, using zeros")
    
    # Fill missing sentiment values
    sentiment_cols = ["sent_mean", "sent_count", "sent_pos", "sent_neg"]
    final_df[sentiment_cols] = final_df[sentiment_cols].fillna(0.0)
    
    # Add other features
    final_df["days_to_earnings"] = 0.0
    
    # Create target variable
    final_df = final_df.sort_values("Date")
    final_df["target"] = final_df["Close"].shift(-horizon_days)
    
    # Select feature columns
    feature_cols = [
        "ret_1", "ret_5", "ma_5", "ma_20", "ma_ratio", "rsi_14", "VIX",
        "sent_mean", "sent_count", "sent_pos", "sent_neg", "days_to_earnings"
    ]
    
    # Create final dataset
    model_data = final_df[["Date"] + feature_cols + ["target"]].dropna()
    
    if len(model_data) == 0:
        st.error("No complete records for modeling")
        return None, None, None, None
    
    X = model_data[feature_cols].values
    y = model_data["target"].values
    dates = model_data["Date"].values
    
    st.write(f"✅ Final dataset: {len(X)} samples, {len(feature_cols)} features")
    st.dataframe(model_data.head(5))
    
    return X, y, dates, feature_cols

def train_model(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, horizon_days: int):
    # Build dataset
    tech_features = build_technical_features(price_df)
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    X, y, dates, feature_cols = create_final_dataset(tech_features, daily_sentiment, horizon_days)
    
    if X is None or len(y) < 30:
        st.error(f"Insufficient data for training. Got {len(y) if y is not None else 0} samples, need at least 30")
        return None
    
    st.write("🤖 Training model")
    
    # Split data
    split_point = int(0.8 * len(y))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Train model
    scaler = StandardScaler()
    model = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    model.partial_fit(X_train_scaled, y_train)
    
    # Evaluate
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sigma = np.std(y_test - y_pred, ddof=1) if len(y_test) > 1 else 0.0
    
    # Make prediction
    X_all_scaled = scaler.transform(X)
    y_all_pred = model.predict(X_all_scaled)
    next_prediction = y_all_pred[-1]
    
    st.write("✅ Model training complete")
    
    return {
        "prediction": next_prediction,
        "current_price": float(price_df["Close"].iloc[-1]),
        "mae": mae,
        "r2": r2,
        "sigma": sigma,
        "n_samples": len(X)
    }

def generate_trading_signal(current_price, predicted_price, sigma, horizon):
    price_change = (predicted_price - current_price) / current_price * 100
    
    # Thresholds based on horizon
    if horizon == "INTRADAY":
        threshold = 1.0  # 1%
    elif horizon == "SHORT":
        threshold = 2.0  # 2%
    else:  # LONG
        threshold = 3.0  # 3%
    
    if price_change > threshold:
        action = "BUY"
    elif price_change < -threshold:
        action = "SELL"
    else:
        action = "HOLD"
    
    # Calculate stop loss
    if sigma > 0:
        stop_pct = max(1.0, min(5.0, 200 * sigma / current_price))  # 1-5%
    else:
        stop_pct = 2.0  # Default 2%
    
    if action == "BUY":
        entry_price = current_price
        stop_loss = entry_price * (1 - stop_pct / 100)
        target_price = predicted_price
    elif action == "SELL":
        entry_price = current_price
        stop_loss = entry_price * (1 + stop_pct / 100)
        target_price = predicted_price
    else:
        entry_price = current_price
        stop_loss = None
        target_price = predicted_price
    
    direction = "UP" if price_change > 0.5 else ("DOWN" if price_change < -0.5 else "NEUTRAL")
    
    return action, entry_price, stop_loss, target_price, price_change, direction

def calculate_position_size(entry_price, stop_price, max_capital=15000):
    if entry_price <= 0:
        return 0, 0
    
    if stop_price and stop_price > 0:
        risk_per_share = abs(entry_price - stop_price)
        max_risk = max_capital * 0.01  # 1% risk
        quantity = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
    else:
        quantity = int(max_capital * 0.1 / entry_price)  # 10% of capital
    
    quantity = max(1, min(quantity, int(max_capital / entry_price)))
    total_cost = quantity * entry_price
    
    return quantity, total_cost

# Streamlit UI
st.title("🚀 NSE/BSE Stock Predictor")
st.write(f"**Market Status:** {'🟢 OPEN' if is_market_open_now() else '🔴 CLOSED'}")

# Input controls
user_input = st.text_input("Enter stock symbol or name:", value="RELIANCE", help="e.g., RELIANCE, TCS, INFY, or RELIANCE.NS")
horizon = st.selectbox("Trading Horizon:", ["INTRADAY", "SHORT", "LONG"], help="INTRADAY=1 day, SHORT=5 days, LONG=20 days")
lookback = st.slider("Historical data (days):", min_value=90, max_value=730, value=365)

if st.button("🔍 Analyze Stock", type="primary"):
    # Step 1: Resolve ticker
    ticker = resolve_ticker(user_input)
    if not ticker:
        st.stop()
    
    # Step 2: Fetch price data
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback)
    price_data = fetch_price_data(ticker, start_date, end_date)
    
    if price_data.empty:
        st.error("No price data available")
        st.stop()
    
    # Step 3: Fetch and analyze news
    news_headlines = fetch_news_data(ticker, limit=15)
    sentiment_data = analyze_sentiment(news_headlines)
    
    # Step 4: Train model and predict
    horizon_days = {"INTRADAY": 1, "SHORT": 5, "LONG": 20}[horizon]
    model_result = train_model(price_data, sentiment_data, horizon_days)
    
    if model_result is None:
        st.error("Could not train model with available data")
        st.stop()
    
    # Step 5: Generate trading signal
    action, entry, stop_loss, target, price_change, direction = generate_trading_signal(
        model_result["current_price"], model_result["prediction"], model_result["sigma"], horizon
    )
    
    # Step 6: Calculate position size
    quantity, investment = calculate_position_size(entry, stop_loss)
    
    # Display results
    st.success("✅ Analysis Complete!")
    
    # Main recommendation
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
        st.metric("Quantity (₹15K max)", f"{quantity} shares")
    
    # Model performance
    with st.expander("🤖 Model Performance"):
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        perf_col1.metric("Mean Absolute Error", f"₹{model_result['mae']:.2f}")
        perf_col2.metric("R² Score", f"{model_result['r2']:.3f}")
        perf_col3.metric("Training Samples", model_result['n_samples'])
    
    # Price chart
    st.subheader("📈 Price History")
    st.line_chart(price_data.set_index(price_data.index)["Close"])
    
    # Summary details
    st.subheader("📝 Summary")
    st.write(f"**Stock:** {ticker}")
    st.write(f"**Date:** {datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    st.write(f"**Term:** {horizon}")
    st.write(f"**Market Open:** {'YES' if is_market_open_now() else 'NO'}")
    st.write(f"**Investment:** ₹{investment:,.0f} ({quantity} shares)")
    
    st.info("⚠️ This is for educational purposes only. Not financial advice. Always do your own research and consult professionals before trading.")
