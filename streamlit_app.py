# streamlit_app.py
import os
import math
import pytz
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from transformers import pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Data providers
import yfinance as yf
from datetime import date as dte
from nsepy import get_history

# ===============================
# Config & Globals
# ===============================
st.set_page_config(page_title="NSE/BSE Expert Predictor (Single File with NSEpy)", layout="wide")

INDIA_TZ = pytz.timezone("Asia/Kolkata")
FINBERT_MODEL = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")

# Optional earnings API (skip if no key)
EARNINGS_API_URL = os.environ.get("EARNINGS_API_URL", "https://api.api-ninjas.com/v1/earningscalendar")
EARNINGS_API_KEY = os.environ.get("EARNINGS_API_KEY", "")

@st.cache_resource(show_spinner=False)
def get_finbert():
    return pipeline("sentiment-analysis", model=FINBERT_MODEL)

def nse_hours(now=None):
    now = now or datetime.now(INDIA_TZ)
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t, close_t

def is_market_open_now():
    now = datetime.now(INDIA_TZ)
    open_t, close_t = nse_hours(now)
    return now.weekday() < 5 and (now >= open_t) and (now <= close_t)

def guess_tickers(user_query: str):
    q = user_query.strip().upper()
    if q.endswith(".NS") or q.endswith(".BO"):
        return [q]
    return [f"{q}.NS", f"{q}.BO"]

def resolve_valid_ticker(user_query: str):
    # Try NSEpy for NSE symbol first by stripping suffix and using get_history
    candidates = guess_tickers(user_query)
    for t in candidates:
        sym_clean = t.replace(".NS","").replace(".BO","")
        try:
            df = get_history(symbol=sym_clean.lower(),  # NSEpy often expects lowercase symbols
                             start=dte(2024,1,1),
                             end=dte(2024,2,1))
            if df is not None and len(df.dropna()) > 0:
                # Confirm with yfinance to ensure data alignment
                yfd = yf.download(f"{sym_clean}.NS", period="3mo", interval="1d", auto_adjust=True, progress=False)
                if yfd is not None and len(yfd.dropna()) > 10:
                    return f"{sym_clean}.NS"
        except Exception:
            pass
    # Fallback: test via yfinance directly
    for t in candidates:
        try:
            yfd = yf.download(t, period="3mo", interval="1d", auto_adjust=True, progress=False)
            if yfd is not None and len(yfd.dropna()) > 10:
                return t
        except Exception:
            pass
    return None

def fetch_prices_nse_first(symbol: str, start_date: date, end_date: date):
    # Try NSEpy for NSE; fallback to yfinance
    sym_clean = symbol.replace(".NS","").replace(".BO","")
    try:
        df = get_history(symbol=sym_clean.lower(),
                         start=dte(start_date.year, start_date.month, start_date.day),
                         end=dte(end_date.year, end_date.month, end_date.day))
        if df is not None and len(df) > 0:
            out = df.rename(columns={
                "Open":"Open","High":"High","Low":"Low","Close":"Close","VWAP":"VWAP","Volume":"Volume"
            })
            out.index = pd.to_datetime(out.index)
            # Ensure standard columns exist
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in out.columns]
            return out[cols].dropna()
    except Exception:
        pass
    # yfinance fallback
    yf_ticker = symbol if (symbol.endswith(".NS") or symbol.endswith(".BO")) else f"{sym_clean}.NS"
    yfd = yf.download(yf_ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    return yfd.dropna()

def fetch_fundamentals(symbol: str):
    t = yf.Ticker(symbol)
    info = {}
    try:
        info["fast_info"] = t.fast_info
    except Exception:
        pass
    try:
        info["info"] = t.info if hasattr(t, "info") else {}
    except Exception:
        info["info"] = {}
    return info

def fetch_news_headlines(symbol: str, limit: int = 60):
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
        items = []
        for n in news[:limit]:
            items.append({
                "title": n.get("title", ""),
                "pubDate": datetime.fromtimestamp(n.get("providerPublishTime", 0), tz=INDIA_TZ),
                "link": n.get("link", ""),
                "publisher": n.get("publisher", ""),
            })
        return items
    except Exception:
        return []

def score_news_finbert(headlines):
    if not headlines:
        return pd.DataFrame(columns=["pubDate","title","sentiment","score"])
    pipe = get_finbert()
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
    return pd.DataFrame(rows).sort_values("pubDate")

def fetch_earnings_calendar(symbol: str, limit: int = 5):
    if not EARNINGS_API_KEY:
        return pd.DataFrame(columns=["date","ticker","actual_eps","estimated_eps"])
    sym_clean = symbol.replace(".NS","").replace(".BO","")
    params = {"ticker": sym_clean, "limit": limit, "show_upcoming":"true"}
    headers = {"X-Api-Key": EARNINGS_API_KEY}
    try:
        r = requests.get(EARNINGS_API_URL, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=["date","ticker","actual_eps","estimated_eps"])

def fetch_india_vix(start_date: date, end_date: date):
    try:
        vix = get_history(symbol="INDIAVIX",
                          start=dte(start_date.year,start_date.month,start_date.day),
                          end=dte(end_date.year,end_date.month,end_date.day),
                          index=True)
        vix = vix.rename(columns={"Close":"VIX"}).filter(["VIX"])
        vix.index = pd.to_datetime(vix.index)
        return vix
    except Exception:
        return pd.DataFrame(columns=["VIX"])

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def build_features(px: pd.DataFrame, vix: pd.DataFrame):
    d = px.copy()
    d["ret_1"] = d["Close"].pct_change()
    d["ret_5"] = d["Close"].pct_change(5)
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_20"] = d["Close"].rolling(20).mean()
    d["ma_ratio"] = d["ma_5"] / d["ma_20"]
    d["rsi_14"] = compute_rsi(d["Close"], 14)
    # Join VIX
    if vix is not None and "VIX" in vix.columns and len(vix)>0:
        d = d.join(vix, how="left")
        d["VIX"] = d["VIX"].fillna(method="ffill").fillna(d["VIX"].median() if len(d["VIX"].dropna())>0 else 0)
    else:
        d["VIX"] = 0.0
    return d.dropna()

def aggregate_sentiment(sent_df: pd.DataFrame):
    if sent_df is None or len(sent_df)==0:
        return pd.DataFrame(columns=["date","sent_mean","sent_count","sent_pos","sent_neg"])
    df = sent_df.copy()
    df["date"] = df["pubDate"].dt.floor("1D")
    label_to_sign = {"positive": 1, "negative": -1, "neutral": 0}
    df["signed"] = df["sentiment"].str.lower().map(label_to_sign).fillna(0) * df["score"]
    agg = df.groupby("date").agg(
        sent_pos=("signed", lambda s: (s[s>0]).sum()),
        sent_neg=("signed", lambda s: (s[s<0]).sum()),
        sent_mean=("signed", "mean"),
        sent_count=("signed","count"),
    ).reset_index()
    return agg

def earnings_days_feature(ev: pd.DataFrame, price_index: pd.DatetimeIndex):
    if ev is None or len(ev)==0:
        return pd.Series(0, index=price_index, name="days_to_earnings")
    events = ev.copy()
    if "date" in events.columns:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
    next_dates = events["date"].dropna().sort_values().values
    if len(next_dates)==0:
        return pd.Series(0, index=price_index, name="days_to_earnings")
    days = []
    for d in price_index:
        if (next_dates > np.datetime64(d)).any():
            nd = next_dates[next_dates > np.datetime64(d)][0]
            days.append((pd.Timestamp(nd)-pd.Timestamp(d)).days)
        else:
            days.append(0)
    return pd.Series(days, index=price_index, name="days_to_earnings")

def make_dataset(px: pd.DataFrame, sent_daily: pd.DataFrame, earn_df: pd.DataFrame, vix: pd.DataFrame, horizon_days: int):
    tech = build_features(px, vix)
    if len(tech)==0:
        return None, None, None, None
    if sent_daily is not None and "date" in sent_daily.columns:
        sent_daily = sent_daily.set_index("date")
    df = tech.join(sent_daily[["sent_mean","sent_count","sent_pos","sent_neg"]] if sent_daily is not None else None, how="left")
    df["days_to_earnings"] = earnings_days_feature(earn_df, df.index)
    df[["sent_mean","sent_count","sent_pos","sent_neg","days_to_earnings"]] = \
        df[["sent_mean","sent_count","sent_pos","sent_neg","days_to_earnings"]].fillna(0)
    y = df["Close"].shift(-horizon_days)
    feature_cols = ["ret_1","ret_5","ma_5","ma_20","ma_ratio","rsi_14","VIX","sent_mean","sent_count","sent_pos","sent_neg","days_to_earnings"]
    out = pd.concat([df[feature_cols], y.rename("y")], axis=1).dropna()
    if len(out)==0:
        return None, None, None, None
    X = out.drop(columns=["y"]).values
    y = out["y"].values
    idx = out.index
    return X, y, idx, feature_cols

def train_and_predict(px: pd.DataFrame, sent_daily: pd.DataFrame, earn_df: pd.DataFrame, vix: pd.DataFrame, horizon_days: int):
    X, y, idx, cols = make_dataset(px, sent_daily, earn_df, vix, horizon_days)
    if X is None or len(y) < 60:
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
    resid = yte - yhat_te
    sigma = np.std(resid, ddof=1) if len(resid) > 1 else 0.0
    yhat_all = model.predict(scaler.transform(X))
    y_next = yhat_all[-1]
    return {
        "pred": y_next,
        "sigma": sigma,
        "last_close": float(px["Close"].iloc[-1]),
        "mae": mae,
        "r2": r2,
        "features_tail": pd.DataFrame(X[-1:, :], columns=cols, index=[idx[-1]])
    }

def action_from_prediction(last_close: float, pred_price: float, sigma: float, sentiment_row: pd.Series, days_to_earn: float, horizon_label: str):
    exp_move = (pred_price - last_close) / last_close if last_close > 0 else 0.0
    sent_mean = float(sentiment_row.get("sent_mean", 0.0)) if sentiment_row is not None else 0.0
    sent_weight = 1.0 + 0.5 * np.tanh(2.0 * sent_mean)
    earn_penalty = 0.7 if (days_to_earn is not None and days_to_earn <= 3 and days_to_earn >= 0) else 1.0
    score = exp_move * sent_weight * earn_penalty
    band = 0.002 if horizon_label == "INTRADAY" else 0.005
    if score > band:
        action = "BUY"
    elif score < -band:
        action = "SELL"
    else:
        action = "HOLD"
    if sigma and sigma > 0:
        vol_pct = min(0.03, max(0.005, sigma / last_close))
    else:
        vol_pct = 0.01 if horizon_label == "INTRADAY" else (0.02 if horizon_label == "SHORT" else 0.04)
    if action == "BUY":
        entry = last_close
        stoploss = entry * (1 - vol_pct)
        target = pred_price
        exp_pct = (target - entry) / entry
    elif action == "SELL":
        entry = last_close
        stoploss = entry * (1 + vol_pct)
        target = pred_price
        exp_pct = (entry - target) / entry
    else:
        entry = last_close
        stoploss = None
        target = pred_price
        exp_pct = abs((target - entry) / entry) if target else 0.0
    exp_pct_clipped = float(np.clip(exp_pct, -0.2, 0.2))
    direction = "UP" if exp_pct_clipped > 0.002 else ("DOWN" if exp_pct_clipped < -0.002 else "NEUTRAL")
    return action, entry, stoploss, target, exp_pct_clipped, direction

def position_size_15k(entry_price: float, stoploss_price: float, max_capital_inr: float = 15000.0, risk_pct: float = 0.01):
    # Classic 1% risk-per-trade position sizing widely discussed in trading literature [Risk rule context]
    max_risk = max_capital_inr * risk_pct
    if entry_price is None or entry_price <= 0:
        return 1, 0.0, max_risk
    if stoploss_price is None or stoploss_price <= 0:
        qty = max(1, int(max_capital_inr // entry_price))
        total_cost = qty * entry_price
        return qty, min(max_capital_inr, total_cost), max_risk
    risk_per_share = abs(entry_price - stoploss_price)
    if risk_per_share <= 0:
        qty = max(1, int(max_capital_inr // entry_price))
        return qty, min(max_capital_inr, qty * entry_price), max_risk
    qty = int(max_risk // risk_per_share)
    qty = 1 if qty < 1 else qty
    total_cost = qty * entry_price
    if total_cost > max_capital_inr:
        qty = max(1, int(max_capital_inr // entry_price))
        total_cost = qty * entry_price
    return qty, total_cost, max_risk

# ===============================
# UI
# ===============================
st.title("NSE/BSE Expert Trade Predictor — Single File (NSEpy + Sentiment + Events)")

c0, c1, c2 = st.columns([2,1,1])
with c0:
    user_query = st.text_input("Enter NSE/BSE stock or company name (e.g., RELIANCE, TCS, INFY, HDFCBANK)", value="RELIANCE")
with c1:
    horizon_label = st.selectbox("TERM", options=["INTRADAY","SHORT","LONG"], index=0)
with c2:
    lookback_days = st.number_input("Lookback days", min_value=90, max_value=1800, value=365, step=5)

if st.button("Get Prediction"):
    ticker = resolve_valid_ticker(user_query)
    if ticker is None:
        st.error("Could not resolve a valid NSE/BSE ticker. Try exact symbol (e.g., RELIANCE.NS) or company name.")
        st.stop()

    st.write(f"Resolved Ticker: {ticker}")
    horizon_days = 1 if horizon_label == "INTRADAY" else (5 if horizon_label == "SHORT" else 20)
    end_date = date.today()
    start_date = end_date - timedelta(days=int(lookback_days))

    with st.spinner("Fetching market data, VIX, fundamentals, headlines..."):
        px = fetch_prices_nse_first(ticker, start_date, end_date + timedelta(days=1))
        vix = fetch_india_vix(start_date, end_date + timedelta(days=1))
        fnd = fetch_fundamentals(ticker)
        headlines = fetch_news_headlines(ticker, limit=80)
        sent_df = score_news_finbert(headlines)
        earn_df = fetch_earnings_calendar(ticker, limit=5)

    st.subheader("Close Price")
    st.line_chart(px["Close"])

    st.subheader("Recent headlines + sentiment")
    st.dataframe(sent_df.sort_values("pubDate", ascending=False).head(12))

    st.subheader("Upcoming earnings (if any)")
    st.dataframe(earn_df.head(5))

    sent_daily = aggregate_sentiment(sent_df)
    pred = train_and_predict(px, sent_daily, earn_df, vix, horizon_days=horizon_days)

    if pred is None:
        st.error("Not enough data to build a prediction. Try increasing lookback days.")
        st.stop()

    last_close = float(pred["last_close"])
    pred_price = float(pred["pred"])
    sigma = float(pred["sigma"])
    mae = float(pred["mae"])
    r2 = float(pred["r2"])

    latest_date = sent_daily["date"].max() if (sent_daily is not None and "date" in sent_daily.columns and len(sent_daily)>0) else None
    latest_sent_row = sent_daily[sent_daily["date"]==latest_date].iloc[0] if latest_date is not None else None
    earn_days_series = earnings_days_feature(earn_df, px.index)
    days_to_earn = float(earn_days_series.iloc[-1]) if (earn_days_series is not None and len(earn_days_series)>0) else None

    action, entry, stoploss, target, exp_pct, direction = action_from_prediction(
        last_close, pred_price, sigma, latest_sent_row, days_to_earn, horizon_label
    )
    qty, total_cap, max_risk = position_size_15k(entry_price=entry, stoploss_price=stoploss, max_capital_inr=15000.0, risk_pct=0.01)

    st.subheader("Expert-style Recommendation")
    st.write(f"STOCK_NAME (COMPANY_NAME): {ticker}")
    st.write(f"DATE: {datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    st.write(f"TERM: {horizon_label}")
    st.write(f"MARKET IS CURRENTLY OPEN? {'YES' if is_market_open_now() else 'NO'}")
    st.write(f"ACTION: {action}")
    st.write(f"PRICE: {'{:.2f}'.format(entry)}")
    if stoploss is not None:
        st.write(f"STOPLOSS: {'{:.2f}'.format(stoploss)}")
    else:
        st.write("STOPLOSS: N/A (HOLD)")
    st.write(f"PREDICTION_DATE (+{horizon_days}D): { (datetime.now(INDIA_TZ)+timedelta(days=horizon_days)).strftime('%Y-%m-%d') }")
    st.write(f"PREDICTED PRICE: {'{:.2f}'.format(target)}")
    st.write(f"HOW MUCH % WILL GO {direction}: {exp_pct*100:.2f}%")
    st.write(f"HOW MANY QTY TO {'BUY' if action!='SELL' else 'SELL'} (<= ₹15,000): {qty} (approx capital ₹{total_cap:,.0f}, max risk/trade ₹{max_risk:,.0f})")

    # Explain section
    reasons = []
    feats = pred["features_tail"].iloc[0]
    if feats.get("ma_ratio", 1.0) > 1.0:
        reasons.append("Short-term trend above 20-day average (bullish bias).")
    if feats.get("rsi_14", 50) < 35:
        reasons.append("RSI indicates potential oversold rebound.")
    if feats.get("rsi_14", 50) > 65:
        reasons.append("RSI shows overbought conditions; pullback risk.")
    if latest_sent_row is not None:
        sm = latest_sent_row.get("sent_mean", 0.0)
        sc = latest_sent_row.get("sent_count", 0)
        if sm > 0.02 and sc >= 2:
            reasons.append("Recent news sentiment is net positive.")
        elif sm < -0.02 and sc >= 2:
            reasons.append("Recent news sentiment is net negative.")
    if days_to_earn is not None and 0 <= days_to_earn <= 3:
        reasons.append("Earnings nearby; confidence reduced and stop widened.")
    if vix is not None and len(vix)>0 and float(vix.iloc[-1].get("VIX", 0)) > (vix["VIX"].median() if "VIX" in vix.columns and len(vix["VIX"].dropna())>0 else 0):
        reasons.append("Elevated India VIX indicates higher market volatility.")
    if not reasons:
        reasons.append("Model-based projection blended with sentiment, VIX, and earnings proximity.")
    st.subheader("EXPLAIN: WHY BUY/SELL/HOLD")
    for r in reasons:
        st.write(f"- {r}")

    st.subheader("Backtest-style Metrics (Holdout)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("R²", f"{r2:.3f}")
    m3.metric("Residual sigma", f"{sigma:.2f}")

    st.caption("Data: NSEpy for NSE (EOD) and Yahoo Finance fallback. NSEpy provides OHLCV/VWAP/delivery and index/VIX series; some symbols may require lowercase with NSEpy, and derivatives can have zeros on illiquid days as noted in docs. Educational demo, not investment advice.")
