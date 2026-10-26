"""Recompute an exported snapshot without starting the web interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from portfolio.core import Market, Optimizer, Restrictions


def reproduce(snapshot):
    market = Market(snapshot["names"], snapshot["annual_returns"], snapshot["annual_covariance"], snapshot["risk_free_rate"])
    s = snapshot["settings"]
    allow_short = s.get("allow_short", False)
    lo, hi = [], []
    for name in market.names:
        lower, upper = s.get("bounds", {}).get(name, (-20. if allow_short else 0., 100.))
        lo.append((lower if allow_short else max(0., lower))/100)
        hi.append(upper/100)
    groups = []
    if s.get("use_groups", False):
        for group in sorted(set(snapshot["groups"])):
            lower, upper = s.get("groups", {}).get(group, (0., 100.))
            indices = tuple(i for i, g in enumerate(snapshot["groups"]) if g == group)
            groups.append((group, indices, lower/100, upper/100))
    if s.get("include_rf", True):
        market = market.with_risk_free()
        lo.append(s.get("rf_lower", 0.)/100)
        hi.append(s.get("rf_upper", 100.)/100)
    opt = Optimizer(market, Restrictions(np.array(lo), np.array(hi), s.get("gross_limit", 100.)/100, tuple(groups)))
    objective = s.get("objective", "平衡收益与风险")
    if objective == "尽可能降低波动":
        result = opt.minimum_variance()
    elif objective == "达到目标收益":
        result = opt.target_return(s["target"]/100)
    elif objective == "控制风险上限":
        result = opt.under_risk_cap(s["cap"]/100)
    elif objective == "获得最高夏普比率":
        result = opt.maximum_sharpe()
    else:
        result = opt.utility(s.get("aversion", 4.))
    return market, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复现 Portfolio Lab 导出的配置参数与结果。")
    parser.add_argument("snapshot", type=Path, help="allocation_snapshot.json 的路径")
    args = parser.parse_args()
    inputs = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    m, p = reproduce(inputs)
    print(json.dumps({"weights": dict(zip(m.names, p.weights.tolist())), "expected_return": p.expected_return,
                      "volatility": p.volatility, "sharpe": p.sharpe,
                      "matches_export": bool(np.allclose(p.weights, inputs["result"]["weights"], atol=2e-6))},
                     ensure_ascii=False, indent=2))
