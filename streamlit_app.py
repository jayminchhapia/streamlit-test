import os
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta, timezone
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="NSE/BSE Predictor — stable merges & headlines", layout="wide")
INDIA_TZ = pytz.timezone("Asia/Kolkata")

# ---------- Utilities ----------
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
    cands = [q] if (q.endswith(".NS") or q.endswith(".BO")) else [f"{q}.NS", f"{q}.BO"]
    st.write("Candidates:", cands)
    for t in cands:
        try:
            df = yf.download(t, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if df is not None and len(df.dropna()) > 10:
                st.write(f"Resolved: {t}")
                return t
        except Exception as e:
            st.write(f"{t} error: {e}")
    st.error("Could not resolve ticker")
    return None

# ---------- Data fetch ----------
def fetch_prices_yf(ticker: str, start_date: date, end_date: date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        st.error("No price data from yfinance")
        return pd.DataFrame()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df = df[df["Close"] > 0]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    st.write(f"Price rows: {len(df)} | range ₹{df['Close'].min():.2f}–₹{df['Close'].max():.2f}")
    st.dataframe(df.head(8))
    return df

def fetch_news(ticker: str, limit: int = 12):
    try:
        t = yf.Ticker(ticker)
        news_raw = getattr(t, "news", None) or []
    except Exception as e:
        st.write(f"news fetch error: {e}")
        news_raw = []

    items = []
    for n in news_raw[:limit]:
        title = (n.get("title") or "").strip()
        publisher = (n.get("publisher") or "").strip()
        link = (n.get("link") or "").strip()
        ts = n.get("providerPublishTime")
        if not title or not ts or ts <= 0:
            continue
        try:
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(INDIA_TZ)
        except Exception:
            continue
        items.append({"title": title, "publisher": publisher or "Unknown", "link": link or "", "pubDate": pub_dt})

    st.write(f"Headlines with valid timestamps: {len(items)}")
    if items:
        st.subheader("Latest headlines (source and link)")
        for i, it in enumerate(items[:10], 1):
            st.write(f"{i}. {it['title']} — {it['publisher']}")
            if it["link"]:
                st.write(it["link"])
    else:
        st.write("No valid headlines available right now.")

    return items

def score_headlines_finbert(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score","publisher"])
    pipe = get_finbert()
    if pipe is None:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score","publisher"])
    texts = [h["title"] for h in headlines if h.get("title")]
    if not texts:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score","publisher"])
    outputs = pipe(texts, truncation=True)
    rows = []
    j = 0
    for h in headlines:
        if not h.get("title"):
            continue
        o = outputs[j]; j += 1
        rows.append({
            "pubDate": pd.to_datetime(h["pubDate"]).tz_localize(None),
            "title": h["title"],
            "sentiment": o.get("label", "neutral"),
            "score": float(o.get("score", 0.0)),
            "publisher": h.get("publisher", "")
        })
    df = pd.DataFrame(rows).sort_values("pubDate")
    st.write("Sentiment counts:", df["sentiment"].value_counts().to_dict())
    st.dataframe(df.tail(8))
    return df

# ---------- Features & dataset ----------
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_technicals(price_df: pd.DataFrame):
    d = price_df.copy()
    d["ret_1"] = d["Close"].pct_change().clip(-0.2, 0.2)
    d["ret_5"] = d["Close"].pct_change(5).clip(-0.5, 0.5)
    d["sma_5"] = d["Close"].rolling(5).mean()
    d["sma_20"] = d["Close"].rolling(20).mean()
    d["ma_ratio"] = (d["sma_5"] / d["sma_20"]).clip(0.8, 1.2)
    d["rsi"] = compute_rsi(d["Close"]).clip(0, 100)
    med_vol = d["Volume"].rolling(20).median()
    d["vol_ratio"] = (d["Volume"] / med_vol).replace([np.inf, -np.inf], np.nan).clip(0.1, 5.0)
    d["volatility"] = d["ret_1"].rolling(10).std().clip(0, 0.1)
    d = d.dropna()
    d = d.reset_index().rename(columns={"index":"Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    st.write(f"Technical rows: {len(d)}")
    st.dataframe(d[["Date","Close","ret_1","ma_ratio","rsi"]].head(8))
    return d

def aggregate_sentiment_daily(sent_df: pd.DataFrame):
    if sent_df is None or len(sent_df) == 0:
        st.write("No sentiment rows to aggregate")
        return pd.DataFrame(columns=["Date","sent_score","sent_count"])
    df = sent_df.copy()
    df["Date"] = pd.to_datetime(df["pubDate"]).dt.normalize()
    sent_map = {"positive": 1, "negative": -1, "neutral": 0}
    df["sent_num"] = df["sentiment"].str.lower().map(sent_map).fillna(0) * df["score"].fillna(0.0)
    agg = df.groupby("Date", as_index=False).agg(
        sent_score=("sent_num","mean"),
        sent_count=("sent_num","count"),
    )
    agg["Date"] = pd.to_datetime(agg["Date"]).dt.normalize()
    agg = agg.drop_duplicates(subset=["Date"])
    st.write(f"Sentiment daily rows: {len(agg)}")
    st.dataframe(agg.head(8))
    return agg

def safe_merge_on_date(tech_df: pd.DataFrame, sent_daily: pd.DataFrame):
    left = tech_df.copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize()
    if sent_daily is None or len(sent_daily) == 0:
        left["sent_score"] = 0.0
        left["sent_count"] = 0.0
        return left
    right = sent_daily.copy()
    right["Date"] = pd.to_datetime(right["Date"]).dt.normalize()
    right = right.drop_duplicates(subset=["Date"])
    # enforce numeric types
    for c in ["sent_score","sent_count"]:
        if c in right.columns:
            right[c] = pd.to_numeric(right[c], errors="coerce").fillna(0.0)
        else:
            right[c] = 0.0
    # Try merge; if it fails (tz/resolution issues), fallback to reindex alignment
    try:
        merged = pd.merge(left, right[["Date","sent_score","sent_count"]],
                          on="Date", how="left", validate="one_to_one")
    except Exception as e:
        st.write(f"merge failed, using fallback align: {e}")
        r_idx = right.set_index("Date")[["sent_score","sent_count"]]
        r_idx = r_idx[~r_idx.index.duplicated(keep="first")]
        l_idx = left.set_index("Date").sort_index()
        aligned = r_idx.reindex(l_idx.index).fillna(0.0)
        merged = pd.concat([l_idx, aligned], axis=1).reset_index()
        merged = merged.rename(columns={"index":"Date"})
    merged["sent_score"] = merged["sent_score"].fillna(0.0)
    merged["sent_count"] = merged["sent_count"].fillna(0.0)
    return merged

def make_dataset(price_df: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    tech = compute_technicals(price_df)
    merged = safe_merge_on_date(tech, sent_daily)
    merged = merged.sort_values("Date")
    # target as percent change for stability
    future = merged["Close"].shift(-horizon_days)
    merged["target_pct"] = ((future - merged["Close"]) / merged["Close"]).clip(-0.3, 0.3)
    feature_cols = ["ret_1","ret_5","ma_ratio","rsi","vol_ratio","volatility","sent_score","sent_count"]
    dataset = merged[["Date","Close"] + feature_cols + ["target_pct"]].dropna()
    st.write(f"Final dataset rows: {len(dataset)} | features: {len(feature_cols)}")
    st.dataframe(dataset.head(8))
    if len(dataset) < 50:
        return None, None, None, None, None
    X = dataset[feature_cols].values
    y = dataset["target_pct"].values
    dates = dataset["Date"].values
    cur_price = float(dataset["Close"].iloc[-1])
    return X, y, dates, feature_cols, cur_price

# ---------- Model ----------
def train_and_predict(price_df: pd.DataFrame, sent_daily: pd.DataFrame, horizon_days: int):
    res = make_dataset(price_df, sent_daily, horizon_days)
    if res[0] is None:
        return None
    X, y, dates, feature_cols, current_price = res
    split = int(0.8 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    scaler = StandardScaler()
    Xtr_b = scaler.fit_transform(Xtr)
    Xte_b = scaler.transform(Xte)
    model = RandomForestRegressor(
        n_estimators=120, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    model.fit(Xtr_b, ytr)
    yhat_te = model.predict(Xte_b)
    mae = mean_absolute_error(yte, yhat_te)
    r2 = r2_score(yte, yhat_te)
    # next prediction
    next_pct = float(model.predict(scaler.transform(X[-1:].copy()))[0])
    # cap to ±15%
    next_pct = float(np.clip(next_pct, -0.15, 0.15))
    pred_price = current_price * (1 + next_pct)
    return {"pred_price": pred_price, "cur_price": current_price, "pred_pct": next_pct, "mae": mae, "r2": r2, "n": len(X)}

def trading_signal(cur_price, pred_price, pred_pct, horizon_label):
    pct = pred_pct * 100
    if horizon_label == "INTRADAY":
        band = 0.5
        stop = 1.5
    elif horizon_label == "SHORT":
        band = 1.0
        stop = 2.5
    else:
        band = 2.0
        stop = 4.0
    if pct > band:
        action = "BUY"
        stoploss = cur_price * (1 - stop/100)
    elif pct < -band:
        action = "SELL"
        stoploss = cur_price * (1 + stop/100)
    else:
        action = "HOLD"
        stoploss = None
    direction = "UP" if pct > 0.2 else ("DOWN" if pct < -0.2 else "NEUTRAL")
    return action, stoploss, pct, direction

def position_size(entry_price, stop_price, cap=15000):
    if entry_price <= 0:
        return 0, 0.0
    if stop_price:
        risk = abs(entry_price - stop_price)
        max_risk = cap * 0.01
        qty = int(max_risk / risk) if risk > 0 else 0
    else:
        qty = int((cap * 0.05) / entry_price)
    qty = max(1, min(qty, int(cap / entry_price)))
    return qty, qty * entry_price

# ---------- UI ----------
st.title("NSE/BSE Predictor — stable merges & sources")
st.write(f"Market open? {'YES' if is_market_open_now() else 'NO'}")

user_query = st.text_input("Symbol or name (e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
horizon_label = st.selectbox("Horizon", ["INTRADAY","SHORT","LONG"], index=0)
lookback_days = st.slider("Lookback days", 180, 730, 365)

if st.button("Analyze", type="primary"):
    ticker = resolve_ticker(user_query)
    if not ticker:
        st.stop()
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    px = fetch_prices_yf(ticker, start_date, end_date)
    if px.empty:
        st.stop()
    headlines = fetch_news(ticker, limit=12)
    sent_rows = score_headlines_finbert(headlines)
    sent_daily = aggregate_sentiment_daily(sent_rows)
    horizon_days = {"INTRADAY":1,"SHORT":5,"LONG":20}[horizon_label]
    result = train_and_predict(px, sent_daily, horizon_days)
    if result is None:
        st.error("Not enough samples for modeling; increase lookback or pick another symbol.")
        st.stop()
    action, stoploss, exp_pct, direction = trading_signal(result["cur_price"], result["pred_price"], result["pred_pct"], horizon_label)
    qty, invest = position_size(result["cur_price"], stoploss)

    st.subheader("Recommendation")
    st.write(f"STOCK_NAME: {ticker}")
    st.write(f"DATE: {datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    st.write(f"TERM: {horizon_label}")
    st.write(f"MARKET IS CURRENTLY OPEN? {'YES' if is_market_open_now() else 'NO'}")
    st.write(f"ACTION: {action}")
    st.write(f"PRICE: ₹{result['cur_price']:.2f}")
    st.write(f"STOPLOSS: {'₹'+format(stoploss,'.2f') if stoploss else 'N/A'}")
    st.write(f"PREDICTED PRICE: ₹{result['pred_price']:.2f}")
    st.write(f"EXPECTED MOVE: {exp_pct:+.2f}% ({direction})")
    st.write(f"QTY (≤ ₹15,000): {qty} (₹{invest:,.0f})")

    st.subheader("Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{result['mae']*100:.2f}%")
    c2.metric("Test R²", f"{result['r2']:.3f}")
    c3.metric("Samples", result["n"])

    st.subheader("Close Price")
    chart_df = px["Close"].reset_index()
    chart_df.columns = ["Date","Close"]
    st.line_chart(chart_df.set_index("Date"))
