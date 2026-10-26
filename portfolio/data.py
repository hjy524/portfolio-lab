"""Demo assumptions and strictly validated historical-return imports."""
from __future__ import annotations

import numpy as np
import pandas as pd


def demo_market():
    assets = pd.DataFrame({
        "code": ["CN_EQ", "GLOBAL_EQ", "GOV_BOND", "CREDIT", "GOLD", "REIT"],
        "name": ["中国股票", "全球股票", "国债", "信用债", "黄金", "REITs"],
        "group": ["权益", "权益", "债券", "债券", "另类", "另类"],
        "return": [0.085, 0.09, 0.028, 0.045, 0.055, 0.075],
        "volatility": [0.23, 0.19, 0.035, 0.065, 0.16, 0.18],
    })
    factors = np.array([[.8, -.1, .0], [.7, -.05, .1], [-.15, .75, .0],
                        [.1, .65, .0], [-.1, .05, .7], [.55, .15, .15]])
    raw = factors @ factors.T + np.diag([.35, .4, .3, .35, .5, .45])
    correlation = raw / np.sqrt(np.outer(np.diag(raw), np.diag(raw)))
    std = assets.volatility.to_numpy()
    covariance = correlation * np.outer(std, std)
    rng = np.random.default_rng(20261026)
    z = rng.normal(size=(120, len(assets)))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z, rowvar=False))).T
    monthly = z @ np.linalg.cholesky(covariance / 12).T + assets["return"].to_numpy() / 12
    history = pd.DataFrame(monthly, columns=assets.code,
                           index=pd.date_range("2016-01-31", periods=120, freq="M"))
    history.index.name = "date"
    return assets, correlation, history


def estimate_from_returns(frame: pd.DataFrame, periods_per_year: int):
    """Require aligned simple returns in decimals; never silently fill observations."""
    if periods_per_year not in (12, 52, 252):
        raise ValueError("请选择月度、周度或日度频率。")
    frame = frame.copy()
    if "date" not in frame.columns:
        raise ValueError("CSV 第一列需命名为 date，后面每列代表一种资产的简单收益率。")
    if len(frame.columns) < 3 or len(frame.columns) > 13:
        raise ValueError("请提供 2–12 种资产。")
    if frame.columns.duplicated().any():
        raise ValueError("资产列名不能重复。")
    if any(not str(c).strip() or str(c).startswith("Unnamed:") for c in frame.columns):
        raise ValueError("资产列名不能为空。")
    dates = pd.to_datetime(frame.pop("date"), errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("date 列含无效日期或重复日期，请先修正。")
    if not dates.is_monotonic_increasing:
        raise ValueError("请按日期从早到晚排序。")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if len(numeric) < max(24, len(numeric.columns) + 2):
        raise ValueError("至少需要 24 期且多于资产数量的对齐观测值。")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("收益数据含缺失值、非数字或无穷大；请先对齐并清理数据。")
    if (values < -1).any():
        raise ValueError("简单收益率不能小于 -100%。请用小数输入，例如 1% 写为 0.01。")
    if (np.abs(values) > 2).any():
        raise ValueError("单期收益超过 200%，请检查是否把百分数 5 当作了小数 0.05。")
    cov = np.cov(values, rowvar=False, ddof=1) * periods_per_year
    volatility = np.sqrt(np.diag(cov))
    if np.any(volatility < 1e-8):
        raise ValueError("存在恒定收益资产。请将无风险资产在专门的设置中添加。")
    if np.linalg.eigvalsh(cov).min() < 1e-10:
        raise ValueError("样本协方差矩阵接近奇异。请移除重复资产或增加观测数据。")
    assets = pd.DataFrame({"code": numeric.columns, "name": numeric.columns,
                           "group": ["自定义"] * len(numeric.columns),
                           "return": values.mean(axis=0) * periods_per_year,
                           "volatility": volatility})
    correlation = cov / np.outer(volatility, volatility)
    numeric.index = pd.DatetimeIndex(dates)
    numeric.index.name = "date"
    return assets, correlation, numeric
