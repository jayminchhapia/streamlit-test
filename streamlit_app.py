# streamlit_app.py
import os
import pytz
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta
from transformers import pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# NSEpy import
from datetime import date as dte
try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    st.warning("NSEpy not available. Using yfinance only.")

# ===============================
# Configuration & Globals
# ===============================
st.set_page_config(page_title="NSE/BSE Stock Predictor with Debug", layout="wide")
INDIA_TZ = pytz.timezone("Asia/Kolkata")
FINBERT_MODEL = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")
EARNINGS_API_URL = os.environ.get("EARNINGS_API_URL", "https://api.api-ninjas.com/v1/earningscalendar")
EARNINGS_API_KEY = os.environ.get("EARNINGS_API_KEY", "")

@st.cache_resource(show_spinner=False)
def get_finbert():
    try:
        return pipeline("sentiment-analysis", model=FINBERT_MODEL)
    except Exception as e:
        st.warning(f"FinBERT not available: {e}")
        return None

def is_market_open_now():
    now = datetime.now(INDIA_TZ)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now.weekday() < 5 and (now >= open_t) and (now <= close_t)

def guess_tickers(user_query: str):
    q = user_query.strip().upper()
    if q.endswith(".NS") or q.endswith(".BO"):
        return [q]
    return [f"{q}.NS", f"{q}.BO"]

def resolve_valid_ticker(user_query: str):
    candidates = guess_tickers(user_query)
    st.write("🔍 **Ticker Resolution Debug:**")
    st.write(f"Candidates to test: {candidates}")
    
    for t in candidates:
        try:
            # Test with yfinance first as it's more reliable
            df = yf.download(t, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if df is not None and len(df.dropna()) > 10:
                st.write(f"✅ Resolved to: {t} (yfinance test passed)")
                return t
            else:
                st.write(f"❌ {t} - No data from yfinance")
        except Exception as e:
            st.write(f"❌ {t} - Error: {e}")
    
    st.error("Could not resolve any valid ticker")
    return None

def fetch_prices(symbol: str, start_date: date, end_date: date):
    st.write(f"📈 **Fetching Price Data for {symbol}**")
    st.write(f"Period: {start_date} to {end_date}")
    
    # Try NSEpy first for NSE symbols
    if NSEPY_AVAILABLE and symbol.endswith(".NS"):
        sym_clean = symbol.replace(".NS", "")
        try:
            df = get_history(
                symbol=sym_clean.upper(),
                start=dte(start_date.year, start_date.month, start_date.day),
                end=dte(end_date.year, end_date.month, end_date.day)
            )
            if df is not None and len(df) > 0:
                df.index = pd.to_datetime(df.index)
                st.write(f"✅ NSEpy data fetched: {len(df)} rows")
                st.write("NSEpy sample data:")
                st.dataframe(df.head(3))
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            st.write(f"⚠️ NSEpy failed: {e}, falling back to yfinance")
    
    # Fallback to yfinance
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        if df is not None and len(df) > 0:
            st.write(f"✅ yfinance data fetched: {len(df)} rows")
            st.write("yfinance sample data:")
            st.dataframe(df.head(3))
            return df.dropna()
        else:
            st.error("No data returned from yfinance")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"yfinance failed: {e}")
        return pd.DataFrame()

def fetch_news_headlines(symbol: str, limit: int = 20):
    st.write(f"📰 **Fetching News for {symbol}**")
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
        st.write(f"Found {len(news)} news items")
        
        items = []
        for n in news[:limit]:
            items.append({
                "title": n.get("title", ""),
                "pubDate": datetime.fromtimestamp(n.get("providerPublishTime", 0), tz=INDIA_TZ),
                "link": n.get("link", ""),
                "publisher": n.get("publisher", ""),
            })
        
        if items:
            st.write("Sample headlines:")
            for i, item in enumerate(items[:3]):
                st.write(f"{i+1}. {item['title']}")
        
        return items
    except Exception as e:
        st.write(f"⚠️ News fetch failed: {e}")
        return []

def score_news_finbert(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    
    pipe = get_finbert()
    if pipe is None:
        st.write("⚠️ FinBERT not available, skipping sentiment")
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    
    st.write(f"🧠 **Analyzing {len(headlines)} headlines with FinBERT**")
    
    texts = [h["title"] for h in headlines]
    outputs = pipe(texts, truncation=True)
    
    rows = []
    for h, o in zip(headlines, outputs):
        rows.append({
            "pubDate": pd.to_datetime(h["pubDate"]),
            "title": h["title"],
            "sentiment": o["label"],
            "score": o["score"],
        })
    
    df = pd.DataFrame(rows).sort_values("pubDate")
    st.write(f"✅ Sentiment analysis complete")
    
    # Show sentiment summary
    sentiment_counts = df['sentiment'].value_counts()
    st.write("Sentiment distribution:")
    st.write(sentiment_counts)
    
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def build_features(px: pd.DataFrame):
    st.write("🔧 **Building Technical Features**")
    
    d = px.copy()
    d["ret_1"] = d["Close"].pct_change()
    d["ret_5"] = d["Close"].pct_change(5)
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_20"] = d["Close"].rolling(20).mean()
    d["ma_ratio"] = d["ma_5"] / d["ma_20"]
    d["rsi_14"] = compute_rsi(d["Close"], 14)
    
    # Add VIX placeholder
    d["VIX"] = 20.0  # Default VIX value
    
    result = d.dropna()
    st.write(f"✅ Features built: {len(result)} rows, {len(result.columns)} columns")
    st.write("Feature sample:")
    st.dataframe(result[["Close", "ret_1", "ma_ratio", "rsi_14"]].tail(3))
    
    return result

def aggregate_sentiment(sent_df: pd.DataFrame):
    if sent_df is None or len(sent_df) == 0:
        st.write("⚠️ No sentiment data to aggregate")
        return pd.DataFrame(columns=["sent_mean","sent_count","sent_pos","sent_neg"])
    
    st.write(f"📊 **Aggregating Sentiment Data**")
    
    df = sent_df.copy()
    df["date"] = df["pubDate"].dt.floor("1D")
    
    label_to_sign = {"positive": 1, "negative": -1, "neutral": 0}
    df["signed"] = df["sentiment"].str.lower().map(label_to_sign).fillna(0) * df["score"]
    
    agg = df.groupby("date").agg(
        sent_pos=("signed", lambda s: (s[s > 0]).sum()),
        sent_neg=("signed", lambda s: (s[s < 0]).sum()),
        sent_mean=("signed", "mean"),
        sent_count=("signed", "count"),
    ).reset_index()
    
    agg = agg.drop_duplicates(subset=["date"]).set_index("date")
    agg.index = pd.to_datetime(agg.index)
    
    st.write(f"✅ Sentiment aggregated to {len(agg)} daily records")
    if len(agg) > 0:
        st.write("Sentiment sample:")
        st.dataframe(agg.tail(3))
    
    return agg

def make_dataset(px: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    st.write(f"🏗️ **Building Dataset for {horizon_days}-day horizon**")
    
    tech = build_features(px)
    
    if len(tech) == 0:
        st.error("No technical features available")
        return None, None, None, None
    
    # Handle sentiment data joining
    df = tech.copy()
    
    if sent_daily is not None and len(sent_daily) > 0:
        # Ensure no duplicates and proper datetime index
        sent_daily = sent_daily[~sent_daily.index.duplicated(keep='first')]
        sent_daily = sent_daily.loc[~sent_daily.index.isnull()]
        
        # Join with sentiment data
        df = df.join(sent_daily[["sent_mean", "sent_count", "sent_pos", "sent_neg"]], how="left")
        st.write(f"✅ Joined with sentiment data")
    else:
        # Create empty sentiment columns
        for col in ["sent_mean", "sent_count", "sent_pos", "sent_neg"]:
            df[col] = 0.0
        st.write("⚠️ No sentiment data available, using zeros")
    
    # Add earnings placeholder
    df["days_to_earnings"] = 0.0
    
    # Fill NaN values
    sentiment_cols = ["sent_mean", "sent_count", "sent_pos", "sent_neg", "days_to_earnings"]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)
    
    # Create target variable
    df["target"] = df["Close"].shift(-horizon_days)
    
    feature_cols = ["ret_1", "ret_5", "ma_5", "ma_20", "ma_ratio", "rsi_14", "VIX", 
                    "sent_mean", "sent_count", "sent_pos", "sent_neg", "days_to_earnings"]
    
    # Create final dataset
    final_df = df[feature_cols + ["target"]].dropna()
    
    if len(final_df) == 0:
        st.error("No complete records after joining and cleaning")
        return None, None, None, None
    
    X = final_df[feature_cols].values
    y = final_df["target"].values
    idx = final_df.index
    
    st.write(f"✅ Dataset created: {len(X)} samples, {len(feature_cols)} features")
    st.write(f"Target range: {y.min():.2f} to {y.max():.2f}")
    
    return X, y, idx, feature_cols

def train_and_predict(px: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    st.write(f"🤖 **Training Model**")
    
    X, y, idx, cols = make_dataset(px, sent_daily, horizon_days)
    
    if X is None or len(y) < 30:
        st.error(f"Insufficient data for training. Need at least 30 samples, got {len(y) if y is not None else 0}")
        return None
    
    # Split data
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    st.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
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
    sigma = np.std(y_test - y_pred, ddof=1)
    
    st.write(f"✅ Model trained successfully")
    st.write(f"MAE: {mae:.2f}, R²: {r2:.3f}, Sigma: {sigma:.2f}")
    
    # Make final prediction
    X_all_scaled = scaler.transform(X)
    y_pred_all = model.predict(X_all_scaled)
    next_pred = y_pred_all[-1]
    
    return {
        "pred": next_pred,
        "sigma": sigma,
        "last_close": float(px["Close"].iloc[-1]),
        "mae": mae,
        "r2": r2,
        "feature_importance": dict(zip(cols, model.coef_)),
        "n_samples": len(X)
    }

def generate_trading_signal(last_close: float, pred_price: float, sigma: float, horizon_label: str):
    st.write(f"🎯 **Generating Trading Signal**")
    
    # Calculate expected move
    exp_move = (pred_price - last_close) / last_close if last_close > 0 else 0.0
    
    # Define thresholds based on horizon
    if horizon_label == "INTRADAY":
        buy_threshold = 0.01  # 1%
        sell_threshold = -0.01
    elif horizon_label == "SHORT":
        buy_threshold = 0.02  # 2%
        sell_threshold = -0.02
    else:  # LONG
        buy_threshold = 0.05  # 5%
        sell_threshold = -0.05
    
    # Generate signal
    if exp_move > buy_threshold:
        action = "BUY"
        confidence = min(1.0, abs(exp_move) / buy_threshold)
    elif exp_move < sell_threshold:
        action = "SELL" 
        confidence = min(1.0, abs(exp_move) / abs(sell_threshold))
    else:
        action = "HOLD"
        confidence = 0.5
    
    # Calculate stop loss
    if sigma > 0:
        stop_pct = max(0.01, min(0.05, 2 * sigma / last_close))
    else:
        stop_pct = 0.02  # 2% default
    
    if action == "BUY":
        entry = last_close
        stoploss = entry * (1 - stop_pct)
        target = pred_price
    elif action == "SELL":
        entry = last_close
        stoploss = entry * (1 + stop_pct)
        target = pred_price
    else:
        entry = last_close
        stoploss = None
        target = pred_price
    
    exp_pct = exp_move * 100
    direction = "UP" if exp_pct > 0.5 else ("DOWN" if exp_pct < -0.5 else "NEUTRAL")
    
    st.write(f"Signal: {action}, Expected move: {exp_pct:.2f}%, Confidence: {confidence:.2f}")
    
    return action, entry, stoploss, target, exp_pct, direction, confidence

def calculate_position_size(entry_price: float, stoploss_price: float, max_capital: float = 15000.0):
    if entry_price <= 0:
        return 0, 0
    
    if stoploss_price is None or stoploss_price <= 0:
        # No stop loss, use fixed allocation
        qty = int(max_capital * 0.1 / entry_price)  # Use 10% of capital
    else:
        # Risk-based position sizing (1% risk)
        risk_per_share = abs(entry_price - stoploss_price)
        max_risk = max_capital * 0.01  # 1% of capital at risk
        qty = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
    
    qty = max(1, min(qty, int(max_capital / entry_price)))  # At least 1 share, max what we can afford
    total_cost = qty * entry_price
    
    return qty, total_cost

# ===============================
# Streamlit UI
# ===============================
st.title("🚀 NSE/BSE Stock Predictor with Step-by-Step Debug")

st.sidebar.header("Configuration")
user_query = st.sidebar.text_input("Stock Symbol/Name", value="RELIANCE")
horizon_label = st.sidebar.selectbox("Trading Horizon", ["INTRADAY", "SHORT", "LONG"])
lookback_days = st.sidebar.slider("Lookback Days", 90, 730, 365)

# Map horizon to days
horizon_map = {"INTRADAY": 1, "SHORT": 5, "LONG": 20}
horizon_days = horizon_map[horizon_label]

st.write(f"**Analysis for:** {user_query}")
st.write(f"**Horizon:** {horizon_label} ({horizon_days} days)")
st.write(f"**Market Open:** {'YES' if is_market_open_now() else 'NO'}")

if st.button("🔍 Analyze Stock", type="primary"):
    # Step 1: Resolve ticker
    ticker = resolve_valid_ticker(user_query)
    if not ticker:
        st.stop()
    
    # Step 2: Fetch price data
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    px = fetch_prices(ticker, start_date, end_date)
    
    if px.empty:
        st.error("No price data available")
        st.stop()
    
    # Step 3: Fetch and analyze news
    headlines = fetch_news_headlines(ticker, limit=20)
    sent_df = score_news_finbert(headlines)
    sent_daily = aggregate_sentiment(sent_df)
    
    # Step 4: Train model and predict
    prediction = train_and_predict(px, sent_daily, horizon_days)
    
    if prediction is None:
        st.error("Could not generate prediction")
        st.stop()
    
    # Step 5: Generate trading signal
    action, entry, stoploss, target, exp_pct, direction, confidence = generate_trading_signal(
        prediction["last_close"], prediction["pred"], prediction["sigma"], horizon_label
    )
    
    # Step 6: Calculate position size
    qty, total_cost = calculate_position_size(entry, stoploss)
    
    # Display results
    st.success("✅ Analysis Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Results")
        st.metric("Current Price", f"₹{entry:.2f}")
        st.metric("Predicted Price", f"₹{target:.2f}")
        st.metric("Expected Move", f"{exp_pct:+.2f}%")
        st.metric("Direction", direction)
        
    with col2:
        st.subheader("💼 Trading Recommendation")
        st.metric("Action", action)
        if stoploss:
            st.metric("Stop Loss", f"₹{stoploss:.2f}")
        st.metric("Quantity", f"{qty} shares")
        st.metric("Investment", f"₹{total_cost:,.0f}")
    
    # Model performance
    with st.expander("🤖 Model Performance"):
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"₹{prediction['mae']:.2f}")
        col2.metric("R² Score", f"{prediction['r2']:.3f}")
        col3.metric("Samples Used", prediction['n_samples'])
    
    # Charts
    st.subheader("📈 Price Chart")
    st.line_chart(px["Close"])
    
    if len(sent_df) > 0:
        st.subheader("📰 Recent News & Sentiment")
        st.dataframe(sent_df.head(10))
    
    # Explanation
    st.subheader("🎯 Why This Recommendation?")
    reasons = []
    
    if abs(exp_pct) > 2:
        reasons.append(f"Model predicts {abs(exp_pct):.1f}% price movement")
    if prediction['r2'] > 0.1:
        reasons.append(f"Model shows decent predictive power (R² = {prediction['r2']:.2f})")
    if len(sent_df) > 0:
        avg_sentiment = sent_df['score'].mean()
        if sent_df['sentiment'].mode().iloc[0] == 'positive':
            reasons.append("Recent news sentiment is positive")
        elif sent_df['sentiment'].mode().iloc[0] == 'negative':
            reasons.append("Recent news sentiment is negative")
    
    if not reasons:
        reasons.append("Based on technical analysis and model prediction")
    
    for reason in reasons:
        st.write(f"• {reason}")
    
    st.warning("⚠️ This is for educational purposes only. Not financial advice. Trade at your own risk.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Debug Info:**")
st.sidebar.markdown(f"NSEpy Available: {NSEPY_AVAILABLE}")
st.sidebar.markdown(f"FinBERT Available: {get_finbert() is not None}")
