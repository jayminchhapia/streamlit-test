import os
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta, timezone
from transformers import pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="NSE/BSE Predictor — robust merges & headlines", layout="wide")
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
    return [q] if (q.endswith(".NS") or q.endswith(".BO")) else [f"{q}.NS", f"{q}.BO"]

def resolve_valid_ticker(user_query: str):
    candidates = guess_tickers(user_query)
    st.write("Candidates:", candidates)
    for t in candidates:
        try:
            df = yf.download(t, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if df is not None and len(df.dropna()) > 10:
                st.write(f"Resolved: {t}")
                return t
        except Exception as e:
            st.write(f"{t} error: {e}")
    st.error("Could not resolve ticker")
    return None

def fetch_prices_yf(ticker: str, start_date: date, end_date: date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        st.error("No yfinance data")
        return pd.DataFrame()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    # Ensure naive datetime64[ns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    st.write(f"yfinance rows: {len(df)} | columns: {list(df.columns)}")
    st.dataframe(df.head(10))
    return df

def fetch_news(ticker: str, limit: int = 20):
    try:
        t = yf.Ticker(ticker)
        news_raw = t.news or []
    except Exception as e:
        st.write(f"news fetch error: {e}")
        news_raw = []

    items = []
    for n in news_raw[:limit]:
        # Handle missing fields safely
        title = n.get("title") or ""
        publisher = n.get("publisher") or ""
        link = n.get("link") or ""
        ts = n.get("providerPublishTime")
        # Filter invalid timestamps (None or <=0 -> skip to avoid 1970-01-01)
        if ts is None or ts <= 0:
            continue
        pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(INDIA_TZ)
        items.append({"title": title, "publisher": publisher, "link": link, "pubDate": pub_dt})

    st.write(f"Headlines fetched: {len(items)}")
    # Display exact items clearly
    if items:
        st.subheader("Latest headlines")
        for i, it in enumerate(items[:10], 1):
            title = it['title'] or "(no title)"
            pub = it['publisher'] or "(unknown)"
            link = it['link'] or "(no link)"
            st.write(f"{i}. {title} — {pub}")
            if link and link != "(no link)":
                st.write(link)
    else:
        st.write("No valid headlines with proper timestamps available.")

    return items

def score_headlines_finbert(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    pipe = get_finbert()
    if pipe is None:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    texts = [h["title"] for h in headlines if h.get("title")]
    if not texts:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    outputs = pipe(texts, truncation=True)
    rows = []
    j = 0
    for h in headlines:
        title = h.get("title")
        if not title:
            continue
        o = outputs[j]
        j += 1
        rows.append({
            "pubDate": pd.to_datetime(h["pubDate"]).tz_localize(None),
            "title": title,
            "sentiment": o.get("label", "neutral"),
            "score": float(o.get("score", 0.0)),
            "publisher": h.get("publisher", "")
        })
    df = pd.DataFrame(rows).sort_values("pubDate")
    st.write("Sentiment distribution:", df["sentiment"].value_counts().to_dict())
    st.dataframe(df.tail(10))
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def build_technicals(px: pd.DataFrame):
    d = px.copy()
    d["ret_1"] = d["Close"].pct_change()
    d["ret_5"] = d["Close"].pct_change(5)
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_20"] = d["Close"].rolling(20).mean()
    d["ma_ratio"] = d["ma_5"] / d["ma_20"]
    d["rsi_14"] = compute_rsi(d["Close"], 14)
    d["VIX"] = 0.0
    d = d.dropna()
    # Flatten to a Date column
    d = d.reset_index().rename(columns={"index":"Date"})
    d["Date"] = pd.to_datetime(d["Date"]).tz_localize(None)
    st.write(f"Features rows: {len(d)}")
    st.dataframe(d[["Date","Close","ret_1","ma_ratio","rsi_14"]].head(10))
    return d

def aggregate_sentiment_daily(sent_df: pd.DataFrame):
    if sent_df is None or len(sent_df) == 0:
        st.write("No sentiment to aggregate")
        return pd.DataFrame(columns=["Date","sent_mean","sent_count","sent_pos","sent_neg"])
    df = sent_df.copy()
    # Already naive datetime above; normalize to date
    df["Date"] = pd.to_datetime(df["pubDate"]).dt.normalize()
    label_to_sign = {"positive": 1, "negative": -1, "neutral": 0}
    df["signed"] = df["sentiment"].str.lower().map(label_to_sign).fillna(0) * df["score"].fillna(0.0)
    agg = df.groupby("Date", as_index=False).agg(
        sent_pos=("signed", lambda s: float((s[s>0]).sum())),
        sent_neg=("signed", lambda s: float((s[s<0]).sum())),
        sent_mean=("signed", "mean"),
        sent_count=("signed","count")
    )
    # Ensure correct dtypes and uniqueness
    agg["Date"] = pd.to_datetime(agg["Date"]).tz_localize(None)
    agg = agg.drop_duplicates(subset=["Date"])
    st.write(f"Sentiment daily rows: {len(agg)}")
    st.dataframe(agg.head(10))
    return agg

def safe_merge_on_date(tech_df: pd.DataFrame, sent_daily: pd.DataFrame):
    # Both should have a naive Date column
    left = tech_df.copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize()
    if sent_daily is None or len(sent_daily)==0:
        for c in ["sent_mean","sent_count","sent_pos","sent_neg"]:
            left[c] = 0.0
        return left
    right = sent_daily.copy()
    right["Date"] = pd.to_datetime(right["Date"]).dt.normalize()
    right = right.drop_duplicates(subset=["Date"])
    # Safe typed columns
    for c in ["sent_mean","sent_count","sent_pos","sent_neg"]:
        if c in right.columns:
            right[c] = pd.to_numeric(right[c], errors="coerce").fillna(0.0)
        else:
            right[c] = 0.0
    try:
        merged = pd.merge(left, right[["Date","sent_mean","sent_count","sent_pos","sent_neg"]],
                          on="Date", how="left", validate="one_to_one")
    except Exception as e:
        st.write(f"merge failed: {e}")
        # Fallback: align by Date via reindex
        right_idx = right.set_index("Date")[["sent_mean","sent_count","sent_pos","sent_neg"]]
        right_idx = right_idx[~right_idx.index.duplicated(keep="first")]
        merged = left.set_index("Date").sort_index()
        right_aligned = right_idx.reindex(merged.index).fillna(0.0)
        merged = pd.concat([merged, right_aligned], axis=1).reset_index()
        merged = merged.rename(columns={"index":"Date"})
    # Fill NaNs after merge
    for c in ["sent_mean","sent_count","sent_pos","sent_neg"]:
        merged[c] = merged[c].fillna(0.0)
    return merged

def make_dataset(px: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    tech = build_technicals(px)  # has Date col
    merged = safe_merge_on_date(tech, sent_daily)
    # Add earnings placeholder
    merged["days_to_earnings"] = 0.0
    # Sort
    merged = merged.sort_values("Date")
    # Target
    merged["target"] = merged["Close"].shift(-horizon_days)
    feature_cols = ["ret_1","ret_5","ma_5","ma_20","ma_ratio","rsi_14","VIX","sent_mean","sent_count","sent_pos","sent_neg","days_to_earnings"]
    final_df = merged[["Date"] + feature_cols + ["target"]].dropna()
    st.write(f"Final dataset rows: {len(final_df)} | features: {len(feature_cols)}")
    st.dataframe(final_df.head(10))
    if len(final_df) == 0:
        return None, None, None, None
    X = final_df[feature_cols].values
    y = final_df["target"].values
    idx = final_df["Date"].values
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

st.title("NSE/BSE Stock Predictor — robust merges & sources")
st.write(f"Market open? {'YES' if is_market_open_now() else 'NO'}")

user_query = st.text_input("Symbol or name (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
horizon_label = st.selectbox("Horizon", ["INTRADAY","SHORT","LONG"], index=0)
lookback_days = st.slider("Lookback days", 90, 730, 365)

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
        st.error("Insufficient data; increase lookback or pick a different ticker.")
        st.stop()
    action, entry, stoploss, target, exp_pct, direction = generate_signal(
        pred["last_close"], pred["pred"], pred["sigma"], horizon_label
    )
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
