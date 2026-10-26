"""Public Chinese ETF history with provenance and an explicit verified snapshot.

Live data use Yahoo adjusted closes, with Eastmoney as an alternate source.
No synthetic fallback is used. Only completed Shanghai trading dates are used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data import estimate_from_returns

SHANGHAI = timezone(timedelta(hours=8))
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data"
ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
YAHOO_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/"
ETF_UNIVERSE = [
    {"code": "510300", "name": "沪深300ETF华泰柏瑞", "group": "权益", "exposure": "A股大盘股票", "exchange": "上交所"},
    {"code": "510500", "name": "中证500ETF南方", "group": "权益", "exposure": "A股中盘股票", "exchange": "上交所"},
    {"code": "513500", "name": "标普500ETF博时", "group": "权益", "exposure": "美国股票，含汇率及溢价风险", "exchange": "上交所"},
    {"code": "511010", "name": "国债ETF国泰", "group": "债券", "exposure": "国债，价格会随利率变化", "exchange": "上交所"},
    {"code": "511260", "name": "十年国债ETF国泰", "group": "债券", "exposure": "长期国债，对利率更敏感", "exchange": "上交所"},
    {"code": "518880", "name": "黄金ETF华安", "group": "另类", "exposure": "黄金，价格也可能明显下跌", "exchange": "上交所"},
]


def shanghai_today():
    return datetime.now(SHANGHAI).date()


def parse_eastmoney(payload, code, start: date, end: date):
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data or str(data.get("code")) != code or not data.get("klines"):
        raise ValueError(f"{code} 没有返回可验证的行情，请稍后重试。")
    rows = []
    for line in data["klines"]:
        parts = line.split(",")
        if len(parts) < 3:
            raise ValueError(f"{code} 的行情格式发生变化。")
        rows.append((parts[0], parts[2]))
    frame = pd.DataFrame(rows, columns=["date", "close"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError(f"{code} 的日期无效或重复。")
    values = frame["close"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{code} 存在无效价格。")
    frame = frame.sort_values("date")
    frame = frame[(frame.date.dt.date >= start) & (frame.date.dt.date <= end)]
    if len(frame) < 121:
        raise ValueError(f"{code} 的有效价格不足 121 期，不能完成当前估计。")
    return frame.set_index("date")["close"].rename(code)


def fetch_etf_prices(code, start: date, end: date):
    params = {"secid": "1."+code, "klt": "101", "fqt": "1", "beg": start.strftime("%Y%m%d"),
              "end": end.strftime("%Y%m%d"), "fields1": "f1,f2,f3,f4,f5,f6",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"}
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.6,
            status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
        response = session.get(ENDPOINT, params=params, timeout=(5, 12),
                                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
    response.raise_for_status()
    return parse_eastmoney(response.json(), code, start, end)


def parse_yahoo(payload, code, start: date, end: date):
    chart = payload.get("chart", {})
    results = chart.get("result") or []
    if chart.get("error") or not results:
        raise ValueError(f"{code} 没有返回可用的复权历史。")
    record = results[0]
    meta = record.get("meta", {})
    if meta.get("symbol") != code+".SS" or meta.get("currency") != "CNY":
        raise ValueError(f"{code} 返回的证券代码或计价币种不符。")
    timestamps = record.get("timestamp", [])
    adjusted = record.get("indicators", {}).get("adjclose", [])
    values = adjusted[0].get("adjclose", []) if adjusted else []
    if not timestamps or len(timestamps) != len(values):
        raise ValueError(f"{code} 未返回完整的复权收盘价，不使用未复权价格替代。")
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    series = pd.Series(values, index=index, name=code, dtype=float).dropna().sort_index()
    if series.index.has_duplicates or not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
        raise ValueError(f"{code} 的复权价格无效。")
    series = series[(series.index.date >= start) & (series.index.date <= end)]
    if len(series) < 121:
        raise ValueError(f"{code} 的完整价格历史不足 121 期。")
    return series


def fetch_yahoo_prices(code, start: date, end: date):
    def stamp(day):
        return int(datetime(day.year, day.month, day.day, tzinfo=SHANGHAI).timestamp())
    params = {"period1": stamp(start), "period2": stamp(end+timedelta(days=1)), "interval": "1d",
              "events": "div,splits", "includeAdjustedClose": "true"}
    response = requests.get(YAHOO_ENDPOINT+code+".SS", params=params,
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=(5, 12))
    response.raise_for_status()
    return parse_yahoo(response.json(), code, start, end)


def build_market(prices: pd.DataFrame, metadata: dict):
    expected = [a["code"] for a in ETF_UNIVERSE]
    prices = prices.copy()
    if list(prices.columns) != expected:
        raise ValueError("行情中的资产代码与候选池不一致。")
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors="raise")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("行情日期必须唯一且递增。")
    if not np.isfinite(prices.to_numpy(float)).all() or (prices <= 0).any().any():
        raise ValueError("行情价格存在缺失或非正值。")
    if prices.index[-1].date() >= shanghai_today():
        raise ValueError("仅使用上海日期之前的完整交易日，避免混入盘中价格。")
    if len(prices) < 121:
        raise ValueError("共同对齐历史不足 120 个日收益观测。")
    # Intersect dates BEFORE pct_change; never forward-fill missing quotes.
    returns = prices.pct_change(fill_method=None).iloc[1:]
    assets, correlation, history = estimate_from_returns(returns.reset_index(names="date"), 252)
    assets["name"] = [a["name"] for a in ETF_UNIVERSE]
    assets["group"] = [a["group"] for a in ETF_UNIVERSE]
    meta = {**metadata, "currency": "CNY", "price_basis": metadata.get("price_basis", "东方财富前复权收盘价"),
            "start_date": prices.index[0].date().isoformat(), "end_date": prices.index[-1].date().isoformat(),
            "observations": len(returns), "periods_per_year": 252,
            "estimator": "日简单收益算术均值×252；样本协方差×252",
            "assets": ETF_UNIVERSE}
    return assets, correlation, history, meta


def fetch_live_market(years=3):
    end = shanghai_today()-timedelta(days=1)
    start = end-timedelta(days=round(365.25*years))
    series, failures = None, []
    providers = [("Yahoo Finance 公开历史行情", YAHOO_ENDPOINT, "Yahoo Finance adjusted close（复权收盘价）", fetch_yahoo_prices),
                 ("东方财富公开历史行情", ENDPOINT, "东方财富前复权收盘价", fetch_etf_prices)]
    for provider, endpoint, basis, fetch_function in providers:
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                series = list(pool.map(lambda asset: fetch_function(asset["code"], start, end), ETF_UNIVERSE))
            break
        except (ValueError, requests.RequestException) as exc:
            failures.append(type(exc).__name__)
    if series is None:
        raise ValueError("在线行情源暂时不可用，请使用已验证的真实快照或稍后重试。")
    outer = pd.concat(series, axis=1, join="outer").sort_index()
    prices = outer.dropna(how="any")
    if len(prices) < len(outer)*.95:
        raise ValueError("多项资产交易日缺失较多；请核对数据源后重试，未自动填补价格。")
    if len(prices) and (end-prices.index[-1].date()).days > 15:
        raise ValueError("接口返回的数据截止日过早，不能标为最新行情。")
    meta = {"provider": provider, "endpoint": endpoint, "price_basis": basis,
            "retrieved_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            "delivery": "在线获取", "dropped_unaligned_dates": len(outer)-len(prices)}
    return prices, build_market(prices, meta)


def load_market_snapshot():
    csv_path = SNAPSHOT_DIR/"cn_etf_prices.csv"
    meta = json.loads((SNAPSHOT_DIR/"cn_etf_snapshot.json").read_text(encoding="utf-8"))
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != meta["csv_sha256"]:
        raise ValueError("真实行情快照校验失败，请重新获取数据。")
    prices = pd.read_csv(csv_path, index_col="date", parse_dates=True, dtype={a["code"]: float for a in ETF_UNIVERSE})
    return build_market(prices, {**meta, "delivery": "已验证真实行情快照"})
