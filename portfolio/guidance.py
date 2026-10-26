"""Plain-language onboarding rules, kept explicit and reproducible."""
from __future__ import annotations

PROFILES = {
    "更在意少波动": {"cap": 5., "equity_max": 25., "alternative_max": 15., "aversion": 10.,
                    "explanation": "优先降低起伏：给股票和黄金较低的比例上限，并在 5% 年化波动上限内寻找收益较高的方案。"},
    "收益与起伏都考虑": {"cap": 10., "equity_max": 55., "alternative_max": 25., "aversion": 4.,
                         "explanation": "在分散配置的同时保留一定股票空间，按 10% 年化波动上限比较组合。"},
    "愿意接受更大起伏": {"cap": 18., "equity_max": 85., "alternative_max": 35., "aversion": 2.,
                         "explanation": "给股票更大的配置空间，波动上限提高到 18%；这不会保证获得更高的实际收益。"},
}
HORIZONS = {
    "半年内可能用到": {"cash_min": 50., "equity_max": 10.},
    "半年到两年内可能用到": {"cash_min": 20., "equity_max": 40.},
    "两年以上暂时不用": {"cash_min": 0., "equity_max": 100.},
}
GROUP_GUIDANCE = {
    "权益": "股票 ETF：同时投资一篮子股票，收益和价格起伏通常更明显；增加比例意味着更依赖股票市场表现。",
    "债券": "债券 ETF：用于分散股票风险，但交易价格仍会涨跌；长期国债对利率变化尤其敏感，并非现金。",
    "另类": "黄金等资产：与股票、债券的表现可能不同，有助于分散，但自身价格也可能明显下跌。",
}


def beginner_settings(names, groups, profile, horizon, cash_min, budget, rf_rate=2.):
    p, h = PROFILES[profile], HORIZONS[horizon]
    reserve = max(float(cash_min), h["cash_min"])
    equity_max = min(p["equity_max"], h["equity_max"])
    group_bounds = {group: [0., equity_max if group == "权益" else p["alternative_max"] if group == "另类" else 100.]
                    for group in set(groups)}
    return {"mode": "普通用户", "objective": "控制风险上限", "aversion": p["aversion"], "target": 6.,
            "cap": p["cap"], "budget": float(budget), "include_rf": True, "allow_short": False,
            "rf_lower": reserve, "rf_upper": 100., "rf_rate": float(rf_rate), "gross_limit": 100.,
            "use_groups": True, "groups": group_bounds, "bounds": {name: [0., 40.] for name in names},
            "beginner_profile": profile, "beginner_horizon": horizon}
