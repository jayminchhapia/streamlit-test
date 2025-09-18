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

st.set_page_config(page_title="NSE/BSE Predictor – robust merge", layout="wide")
INDIA_TZ = pytz.timezone("Asia/Kolkata")
FINBERT_MODEL = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")

@st.cache_resource(show_spinner=False)
def get_finbert():
    try:
        return pipeline("sentiment-analysis", model=FINBERT_MODEL)
    except Exception:
        return None

def is_market_open_now():
    now = datetime.now(INDIA_TZ)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now.weekday() < 5 and (now >= open_t) and (now <= close_t)

def guess_tickers(q: str):
    q = q.strip().upper()
    if q.endswith(".NS") or q.endswith(".BO"):
        return [q]
    return [f"{q}.NS", f"{q}.BO"]

def resolve_valid_ticker(user_query: str):
    candidates = guess_tickers(user_query)
    st.write("🔍 Ticker resolution candidates:", candidates)
    for t in candidates:
        try:
            df = yf.download(t, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if df is not None and len(df.dropna()) > 10:
                st.write(f"✅ Resolved to: {t}")
                return t
        except Exception as e:
            st.write(f"❌ {t}: {e}")
    st.error("Could not resolve any valid ticker")
    return None

def fetch_prices_yf(ticker: str, start_date: date, end_date: date):
    st.write(f"📈 Fetching yfinance OHLCV for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or len(df)==0:
        st.error("yfinance returned no data")
        return pd.DataFrame()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    st.write(f"✅ yfinance rows: {len(df)}")
    st.dataframe(df.head(3))
    return df

def fetch_news(ticker: str, limit: int = 20):
    st.write(f"📰 Fetching news for {ticker}")
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        items = []
        for n in news[:limit]:
            items.append({
                "title": n.get("title", ""),
                "pubDate": datetime.fromtimestamp(n.get("providerPublishTime", 0), tz=INDIA_TZ),
                "link": n.get("link", ""),
                "publisher": n.get("publisher", "")
            })
        st.write(f"Found {len(items)} headlines")
        if items:
            # Show exact 10 headlines with source
            st.subheader("Latest headlines (title, publisher, link)")
            for i, it in enumerate(items[:10], 1):
                st.write(f"{i}. {it['title']} — {it['publisher']}")
                st.write(it["link"])
        return items
    except Exception as e:
        st.write(f"⚠️ News fetch failed: {e}")
        return []

def score_headlines_finbert(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    pipe = get_finbert()
    if pipe is None:
        st.write("⚠️ FinBERT unavailable; skipping sentiment")
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    texts = [h["title"] for h in headlines]
    outputs = pipe(texts, truncation=True)
    rows = []
    for h, o in zip(headlines, outputs):
        rows.append({
            "pubDate": pd.to_datetime(h["pubDate"]),
            "title": h["title"],
            "sentiment": o["label"],
            "score": o["score"]
        })
    df = pd.DataFrame(rows).sort_values("pubDate")
    st.write("Sentiment distribution:", df["sentiment"].value_counts().to_dict())
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def build_technicals(px: pd.DataFrame):
    st.write("🔧 Building technical features")
    d = px.copy()
    d["ret_1"] = d["Close"].pct_change()
    d["ret_5"] = d["Close"].pct_change(5)
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_20"] = d["Close"].rolling(20).mean()
    d["ma_ratio"] = d["ma_5"] / d["ma_20"]
    d["rsi_14"] = compute_rsi(d["Close"], 14)
    d["VIX"] = 0.0  # placeholder
    d = d.dropna()
    # Flatten index for robust merges
    d = d.reset_index().rename(columns={"index":"Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
    st.write(f"✅ Features rows: {len(d)}")
    st.dataframe(d[["Date","Close","ret_1","ma_ratio","rsi_14"]].tail(3))
    return d

def aggregate_sentiment_daily(sent_df: pd.DataFrame):
    if sent_df is None or len(sent_df)==0:
        st.write("⚠️ No sentiment to aggregate")
        return pd.DataFrame(columns=["Date","sent_mean","sent_count","sent_pos","sent_neg"])
    df = sent_df.copy()
    df["Date"] = df["pubDate"].dt.floor("1D").dt.tz_localize(None)
    label_to_sign = {"positive": 1, "negative": -1, "neutral": 0}
    df["signed"] = df["sentiment"].str.lower().map(label_to_sign).fillna(0) * df["score"]
    agg = df.groupby("Date").agg(
        sent_pos=("signed", lambda s: (s[s>0]).sum()),
        sent_neg=("signed", lambda s: (s[s<0]).sum()),
        sent_mean=("signed","mean"),
        sent_count=("signed","count")
    ).reset_index()
    agg = agg.drop_duplicates(subset=["Date"])
    st.write(f"✅ Sentiment daily rows: {len(agg)}")
    st.dataframe(agg.tail(3))
    return agg

def make_dataset(px: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    st.write(f"🏗️ Building dataset (horizon={horizon_days}d)")
    tech = build_technicals(px)  # has Date column
    if sent_daily is None or len(sent_daily)==0:
        # Build empty sentiment columns
        tech["sent_mean"] = 0.0
        tech["sent_count"] = 0.0
        tech["sent_pos"] = 0.0
        tech["sent_neg"] = 0.0
        merged = tech.copy()
    else:
        # Safe merge on Date column, not index
        sent_daily = sent_daily.copy()
        sent_daily["Date"] = pd.to_datetime(sent_daily["Date"])
        merged = pd.merge(
            tech, 
            sent_daily[["Date","sent_mean","sent_count","sent_pos","sent_neg"]],
            on="Date", how="left"
        )
        for c in ["sent_mean","sent_count","sent_pos","sent_neg"]:
            merged[c] = merged[c].fillna(0)
    # Add earnings placeholder
    merged["days_to_earnings"] = 0.0
    # Target
    merged = merged.sort_values("Date")
    merged["target"] = merged["Close"].shift(-horizon_days)
    feature_cols = ["ret_1","ret_5","ma_5","ma_20","ma_ratio","rsi_14","VIX","sent_mean","sent_count","sent_pos","sent_neg","days_to_earnings"]
    final_df = merged[["Date"] + feature_cols + ["target"]].dropna()
    if len(final_df)==0:
        st.error("No samples after cleaning; increase lookback or change ticker")
        return None, None, None, None
    X = final_df[feature_cols].values
    y = final_df["target"].values
    idx = final_df["Date"].values
    st.write(f"✅ Dataset: {len(X)} samples, {len(feature_cols)} features")
    return X, y, idx, feature_cols

def train_and_predict(px: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    X, y, idx, cols = make_dataset(px, sent_daily, horizon_days)
    if X is None or len(y) < 40:
        return None
    split = int(0.8 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    scaler = StandardScaler()
    model = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4, random_state=42)
    Xtr_b = scaler.fit_transform(Xtr)
    model.partial_fit(Xtr_b, ytr)
    yhat_te = model.predict(scaler.transform(Xte))
    mae = mean_absolute_error(yte, yhat_te)
    r2 = r2_score(yte, yhat_te)
    sigma = np.std(yte - yhat_te, ddof=1) if len(yte) > 1 else 0.0
    yhat_all = model.predict(scaler.transform(X))
    next_pred = yhat_all[-1]
    return {"pred": next_pred, "sigma": sigma, "last_close": float(px["Close"].iloc[-1]),
            "mae": mae, "r2": r2, "n_samples": len(X)}

def generate_signal(last_close, pred_price, sigma, horizon_label):
    exp_move = (pred_price - last_close) / last_close if last_close > 0 else 0.0
    band = 0.01 if horizon_label=="INTRADAY" else (0.02 if horizon_label=="SHORT" else 0.05)
    if exp_move > band:
        action = "BUY"
    elif exp_move < -band:
        action = "SELL"
    else:
        action = "HOLD"
    stop_pct = max(0.01, min(0.05, 2*sigma/last_close)) if sigma>0 else 0.02
    if action=="BUY":
        entry=last_close; stoploss=entry*(1-stop_pct); target=pred_price
    elif action=="SELL":
        entry=last_close; stoploss=entry*(1+stop_pct); target=pred_price
    else:
        entry=last_close; stoploss=None; target=pred_price
    exp_pct = exp_move*100
    direction = "UP" if exp_pct>0.5 else ("DOWN" if exp_pct<-0.5 else "NEUTRAL")
    return action, entry, stoploss, target, exp_pct, direction

def position_size(entry: float, stop: float|None, cap: float=15000.0):
    if entry<=0: return 0, 0.0
    if stop:
        rps = abs(entry-stop); max_risk = 0.01*cap; qty = int(max_risk/rps) if rps>0 else 0
    else:
        qty = int(0.1*cap/entry)
    qty = max(1, min(qty, int(cap/entry)))
    return qty, qty*entry

st.title("🚀 NSE/BSE Stock Predictor — yfinance-only with robust column merges")
st.write(f"Market open now? {'YES' if is_market_open_now() else 'NO'}")

user_query = st.text_input("Stock symbol or name (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
horizon_label = st.selectbox("Trading Horizon", ["INTRADAY","SHORT","LONG"], index=0)
lookback_days = st.slider("Lookback Days", 90, 730, 365)

if st.button("Analyze", type="primary"):
    ticker = resolve_valid_ticker(user_query)
    if not ticker:
        st.stop()
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    px = fetch_prices_yf(ticker, start_date, end_date)
    if px.empty:
        st.stop()
    headlines = fetch_news(ticker, limit=20)
    sent_rows = score_headlines_finbert(headlines)
    sent_daily = aggregate_sentiment_daily(sent_rows)
    horizon_days = {"INTRADAY":1,"SHORT":5,"LONG":20}[horizon_label]
    pred = train_and_predict(px, sent_daily, horizon_days)
    if pred is None:
        st.error("Insufficient data for training. Increase lookback or pick another ticker.")
        st.stop()
    action, entry, stoploss, target, exp_pct, direction = generate_signal(pred["last_close"], pred["pred"], pred["sigma"], horizon_label)
    qty, total = position_size(entry, stoploss)
    st.subheader("Recommendation")
    st.write(f"STOCK_NAME: {ticker}")
    st.write(f"DATE: {datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    st.write(f"TERM: {horizon_label}")
    st.write(f"MARKET IS CURRENTLY OPEN? {'YES' if is_market_open_now() else 'NO'}")
    st.write(f"ACTION: {action}")
    st.write(f"PRICE: ₹{entry:.2f}")
    st.write(f"STOPLOSS: {'₹'+format(stoploss,'.2f') if stoploss else 'N/A'}")
    st.write(f"PREDICTED PRICE: ₹{target:.2f}")
    st.write(f"EXPECTED MOVE: {exp_pct:+.2f}% ({direction})")
    st.write(f"QTY (≤ ₹15,000): {qty} (₹{total:,.0f})")
    st.subheader("Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"₹{pred['mae']:.2f}")
    c2.metric("R²", f"{pred['r2']:.3f}")
    c3.metric("Samples", pred["n_samples"])
    st.subheader("Close Price")
    st.line_chart(px["Close"])
