"""Consistent summaries across the web page and downloadable reports."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weight_details(names, portfolio, metadata, budget):
    records = []
    for name, weight in zip(names, portfolio.weights):
        info = metadata.get(name, {})
        records.append({"资产": name, "证券／资产代码": info.get("code", "未填写"),
                        "类别": info.get("group", "自定义"), "配置比例 (%)": float(weight*100),
                        "配置金额（元）": float(weight*budget)})
    return pd.DataFrame(records)


def largest_holding(names, portfolio, metadata):
    # Highest absolute exposure is meaningful for both long-only and shorts.
    i = int(np.argmax(np.abs(portfolio.weights)))
    name = names[i]
    return {"name": name, "code": metadata.get(name, {}).get("code", "未填写"),
            "weight": float(portfolio.weights[i]), "group": metadata.get(name, {}).get("group", "自定义")}


def one_line(names, portfolio, metadata, kind="我的方案"):
    top = largest_holding(names, portfolio, metadata)
    if portfolio.volatility < 1e-9:
        return f"{kind}全部配置于模型无风险资产，预期年收益 {portfolio.expected_return:.2%}、模型波动为零；ETF 国债基金本身不属于这里的无风险资产。"
    role = {"最小方差方案": "在当前持仓限制下优先降低波动",
            "仅风险资产最小方差": "在不使用现金的前提下优先降低波动",
            "切线／最大夏普风险组合": "在风险资产中追求最高的单位波动超额收益"}.get(kind, "按你的目标与限制计算")
    side = "空头" if top["weight"] < 0 else "持仓"
    return (f"{role}，预期年收益 {portfolio.expected_return:.2%}、年化波动 {portfolio.volatility:.2%}；"
            f"最大{side}为 {top['name']}（{top['code']}），比例 {top['weight']:.1%}。")


def comparison_table(plans, metadata):
    rows = []
    for label, names, p in plans:
        if p is None:
            continue
        top = largest_holding(names, p, metadata)
        rows.append({"方案": label, "预期年收益 (%)": p.expected_return*100,
                     "年化波动 (%)": p.volatility*100, "夏普比率": p.sharpe,
                     "最大仓位资产": top["name"], "证券／资产代码": top["code"],
                     "最大仓位 (%)": top["weight"]*100})
    return pd.DataFrame(rows)


def weights_comparison(plans, metadata, categories=False):
    rows = {}
    for label, names, p in plans:
        if p is None:
            continue
        for name, weight in zip(names, p.weights):
            info = metadata.get(name, {})
            key = info.get("group", "自定义") if categories else name
            if key not in rows:
                rows[key] = {"类别" if categories else "资产": key}
                if not categories:
                    rows[key].update({"证券／资产代码": info.get("code", "未填写"), "类别": info.get("group", "自定义")})
            rows[key][label+" (%)"] = rows[key].get(label+" (%)", 0.) + float(weight*100)
    return pd.DataFrame(rows.values()).fillna(0.)
