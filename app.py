"""Run with: python -m streamlit run app.py"""
from __future__ import annotations

import hashlib
import csv
import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio.core import AllocationError, Market, Optimizer, Restrictions, cal_interval
from portfolio.data import demo_market, estimate_from_returns

st.set_page_config(page_title="配比 Portfolio Lab · 资产配置工具箱", page_icon="◒", layout="wide")

COLORS = ["#147D72", "#477EA8", "#D8A44D", "#8696C5", "#AF7569", "#5CA6A6", "#9BAA65", "#6B879B"]
STEPS = ["01  选择资产", "02  设置目标与限制", "03  查看配置方案", "04  方法与作业对照"]


def style():
    st.markdown("""<style>
    .block-container {padding-top:2.4rem; padding-bottom:3rem; max-width:1450px;}
    h1,h2,h3 {letter-spacing:-.035em;}
    h1 {font-size:2.65rem !important; font-weight:700 !important;}
    h2 {font-size:1.55rem !important;}
    h3 {font-size:1.1rem !important;}
    [data-testid="stSidebar"] {background:#EDF2F5;}
    [data-testid="stMetric"] {background:white; border:1px solid #E2E9EE;
        border-radius:14px; padding:1rem 1.2rem;}
    [data-testid="stMetricLabel"] {color:#6E7C88;}
    [data-testid="stMetricValue"] {font-size:1.8rem;}
    .eyebrow {font-size:.77rem;letter-spacing:.17em;color:#147D72;font-weight:700;margin-bottom:10px;}
    .intro {font-size:1.03rem;color:#607383;max-width:800px;line-height:1.85;}
    .note {background:#EAF4F1;border:1px solid #D5E7E1;padding:17px 21px;
        border-radius:12px;color:#335F57;line-height:1.8;margin:15px 0 22px;}
    .foot {font-size:.78rem;color:#7A8B96;line-height:1.9;padding-top:1rem;}
    div.stButton > button {border-radius:9px;min-height:2.65rem;}
    </style>""", unsafe_allow_html=True)


def navigate(step):
    st.session_state["navigation"] = STEPS[step]


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def chart_layout(fig, height=440):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#365062"},
                      margin={"l": 15, "r": 15, "t": 25, "b": 25},
                      legend={"orientation": "h", "y": -0.2}, hovermode="closest")
    fig.update_xaxes(gridcolor="#E5EBF0", zeroline=False)
    fig.update_yaxes(gridcolor="#E5EBF0", zeroline=False)
    return fig


def get_market_inputs():
    assets, correlation, history = demo_market()
    source = st.session_state.get("source_choice", st.session_state.get("data_source", "内置模拟资产"))
    if source == "导入历史收益 CSV":
        if "imported_market" not in st.session_state:
            raise AllocationError("请先在第一步上传有效的历史收益 CSV，或切回内置模拟资产。")
        assets, correlation, history = st.session_state["imported_market"]
    base = pd.DataFrame({"启用": True, "资产名称": assets["name"], "类别": assets["group"],
                         "预期年收益 (%)": assets["return"]*100,
                         "年化波动 (%)": assets["volatility"]*100})
    key = signature({"assets": assets.to_json(), "corr": correlation.tolist()})[:12]
    return base, correlation, history, key, source


def selected_market():
    base, corr, history, data_key, source = get_market_inputs()
    edited = st.session_state.get("asset_values_"+data_key, base).copy()
    edited = edited.dropna(how="all")
    selected = edited[edited["启用"].fillna(False).astype(bool)].copy()
    if len(selected) < 2 or len(selected) > 12:
        raise AllocationError("请启用 2–12 项风险资产。")
    names = selected["资产名称"].fillna("").astype(str).str.strip().tolist()
    if any(not name for name in names) or len(set(names)) != len(names) or "无风险资产" in names:
        raise AllocationError("资产名称不能为空、不能重复，且请将无风险资产在第二步单独添加。")
    try:
        mu = pd.to_numeric(selected["预期年收益 (%)"], errors="raise").to_numpy(float)/100
        vol = pd.to_numeric(selected["年化波动 (%)"], errors="raise").to_numpy(float)/100
    except (TypeError, ValueError):
        raise AllocationError("请完整填写每项资产的收益和波动数值。") from None
    if not np.isfinite(vol).all() or np.any(vol <= 0):
        raise AllocationError("风险资产的年化波动必须大于 0。")
    full_corr = np.eye(len(edited))
    for i, original_i in enumerate(edited.index):
        for j, original_j in enumerate(edited.index):
            if original_i in base.index and original_j in base.index:
                full_corr[i, j] = corr[base.index.get_loc(original_i), base.index.get_loc(original_j)]
    corr_key = "correlation_"+signature({"data": data_key, "names": edited["资产名称"].fillna("").tolist()})[:12]
    custom = st.session_state.get(corr_key)
    if custom is not None and np.asarray(custom).shape == full_corr.shape:
        full_corr = np.asarray(custom, float)
    if not np.isfinite(full_corr).all() or np.any(np.abs(full_corr) > 1+1e-10) or not np.allclose(np.diag(full_corr), 1):
        raise AllocationError("相关系数需在 -1 到 1 之间，且主对角线必须为 1。")
    indices = [edited.index.get_loc(i) for i in selected.index]
    subset_corr = full_corr[np.ix_(indices, indices)]
    rf = st.session_state.get("settings", {}).get("rf_rate", 2.0)
    market = Market(names, mu, subset_corr*np.outer(vol, vol), st.session_state.get("rf_rate", rf)/100)
    groups = selected["类别"].fillna("自定义").astype(str).tolist()
    return market, groups, source, history, data_key


def page_assets():
    st.markdown('<div class="eyebrow">STEP 01 · INVESTMENT UNIVERSE</div>', unsafe_allow_html=True)
    st.title("先选资产，再谈配置。")
    st.markdown('<p class="intro">告诉工具你愿意投资什么。保留示例即可开始，也可以加入自己的资产与判断。</p>', unsafe_allow_html=True)
    sources = ["内置模拟资产", "导入历史收益 CSV"]
    source = st.radio("数据来源", sources, index=sources.index(st.session_state.get("data_source", sources[0])), horizontal=True, key="source_choice")
    st.session_state["data_source"] = source
    if source == "导入历史收益 CSV":
        with st.container(border=True):
            st.write("**上传对齐后的简单收益率**")
            st.caption("第一列为 date；其余每列是一项资产。1% 请写成 0.01。至少 24 期，2–12 项资产，不接受缺失值。")
            a, b = st.columns([2, 1])
            uploaded = a.file_uploader("选择 CSV 文件", type=["csv"])
            freq_name = b.selectbox("每行数据频率", ["月度（12 期/年）", "周度（52 期/年）", "日度（252 期/年）"])
            freq = {"月度（12 期/年）": 12, "周度（52 期/年）": 52, "日度（252 期/年）": 252}[freq_name]
            _, _, demo_history = demo_market()
            st.download_button("下载格式示例（模拟月度收益）", demo_history.to_csv().encode("utf-8-sig"),
                               "example_monthly_returns.csv", "text/csv")
            if uploaded is not None:
                upload_id = hashlib.sha256(uploaded.getvalue()+str(freq).encode()).hexdigest()
                try:
                    if st.session_state.get("upload_id") != upload_id:
                        header = next(csv.reader(io.StringIO(uploaded.getvalue().decode("utf-8-sig"))), [])
                        if len(header) != len(set(header)):
                            raise ValueError("CSV 资产列名不能重复，请先修改列名。")
                        raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                        imported = estimate_from_returns(raw, freq)
                        st.session_state["imported_market"] = imported
                        st.session_state["upload_id"] = upload_id
                    st.success("导入成功：已用样本均值、样本协方差进行年化估计。你仍可修改下方假设。")
                except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                    st.session_state.pop("imported_market", None)
                    st.session_state.pop("upload_id", None)
                    st.error(f"导入失败：{exc}")
                    return
            elif "imported_market" not in st.session_state:
                st.info("上传文件后即可继续；也可以切回内置模拟资产。")
                return
    else:
        st.markdown('<div class="note">课堂演示数据 · 6 类资产，统一以人民币计价。收益、波动和相关性均为假设，历史序列为人工生成，不代表真实市场表现。</div>', unsafe_allow_html=True)
    base, corr, history, data_key, _ = get_market_inputs()
    left, right = st.columns([3.2, 1])
    with left:
        st.subheader("可投资资产")
        st.caption("勾选启用资产；双击单元格即可修改。底部可新增资产，新增资产默认与其他资产零相关。")
        editor_base_key = "asset_editor_base_"+data_key
        if st.session_state.get("page_changed") or editor_base_key not in st.session_state:
            st.session_state[editor_base_key] = st.session_state.get("asset_values_"+data_key, base).copy()
        edited = st.data_editor(st.session_state[editor_base_key], key="assets_editor_"+data_key, num_rows="dynamic", hide_index=True,
            use_container_width=True,
            column_config={
                "启用": st.column_config.CheckboxColumn(default=True),
                "资产名称": st.column_config.TextColumn(required=True),
                "类别": st.column_config.TextColumn(required=True),
                "预期年收益 (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=200.0, step=.1, format="%.2f"),
                "年化波动 (%)": st.column_config.NumberColumn(min_value=.01, max_value=200.0, step=.1, format="%.2f"),
            })
        st.session_state["asset_values_"+data_key] = edited.copy()
    with right:
        st.subheader("如何填写")
        st.write("**预期收益**\n\n你对未来一年的收益判断。")
        st.write("**年化波动**\n\n收益的不确定程度，也是本工具的风险尺度。")
        st.caption("新增一行时，请完整填写名称、类别、收益和波动，并勾选启用。")
    with st.expander("专业设置 · 查看与编辑相关系数"):
        names = edited["资产名称"].fillna("").astype(str).tolist()
        if names and len(names) == len(set(names)) and all(names) and len(names) <= 12:
            full = np.eye(len(edited))
            for i, original_i in enumerate(edited.index):
                for j, original_j in enumerate(edited.index):
                    if original_i in base.index and original_j in base.index:
                        full[i, j] = corr[base.index.get_loc(original_i), base.index.get_loc(original_j)]
            corr_key = "correlation_"+signature({"data": data_key, "names": names})[:12]
            st.caption("修改非对角元素时，请同时修改对称位置；矩阵须半正定。每个相关系数应在 -1 到 1 之间。")
            corr_base_key = "corr_base_"+corr_key
            if st.session_state.get("page_changed") or corr_base_key not in st.session_state:
                st.session_state[corr_base_key] = pd.DataFrame(st.session_state.get(corr_key, full), index=names, columns=names)
            corr_edited = st.data_editor(st.session_state[corr_base_key],
                                         key="corr_editor_"+corr_key, use_container_width=True,
                                         column_config={name: st.column_config.NumberColumn(min_value=-1., max_value=1., step=.05, format="%.3f") for name in names})
            st.session_state[corr_key] = corr_edited.to_numpy(float)
        else:
            st.info("请先填写唯一且完整的资产名称。")
    try:
        market, _, _, _, _ = selected_market()
        st.caption(f"当前已启用 {len(market.names)} 项资产 · 参数通过基本检查")
        st.button("下一步：设置投资目标 →", type="primary", on_click=navigate, args=(1,))
    except AllocationError as exc:
        st.error(str(exc))


def settings(market, groups):
    n = len(market.names)
    s = st.session_state.get("settings", {})
    include_rf = s.get("include_rf", True)
    full_market = market.with_risk_free() if include_rf else market
    allow_short = s.get("allow_short", False)
    minimum = s.get("bounds", {})
    lower, upper = [], []
    for name in market.names:
        lo, hi = minimum.get(name, (-20.0 if allow_short else 0.0, 100.0))
        if not allow_short:
            lo = max(0., lo)
        lower.append(lo/100)
        upper.append(hi/100)
    if include_rf:
        lower.append(s.get("rf_lower", 0.)/100)
        upper.append(s.get("rf_upper", 100.)/100)
    group_rules = []
    if s.get("use_groups", False):
        for group in sorted(set(groups)):
            lo, hi = s.get("groups", {}).get(group, (0., 100.))
            group_rules.append((group, tuple(i for i, g in enumerate(groups) if g == group), lo/100, hi/100))
    rules = Restrictions(np.asarray(lower), np.asarray(upper), s.get("gross_limit", 100.)/100, tuple(group_rules))
    risky_rules = Restrictions(np.asarray(lower[:n]), np.asarray(upper[:n]), rules.gross_limit, tuple(group_rules))
    return full_market, rules, risky_rules, s


def page_settings():
    st.markdown('<div class="eyebrow">STEP 02 · YOUR INVESTMENT RULES</div>', unsafe_allow_html=True)
    st.title("让组合适合你的要求。")
    st.markdown('<p class="intro">选择目标，再加上必须遵守的规则。所有比例都按总投资资金计算。</p>', unsafe_allow_html=True)
    try:
        market, groups, _, _, data_key = selected_market()
    except AllocationError as exc:
        st.error(str(exc))
        st.button("返回选择资产", on_click=navigate, args=(0,))
        return
    left, right = st.columns([1.15, 1])
    saved = st.session_state.get("settings", {})
    with left, st.container(border=True):
        st.subheader("你希望优先实现什么？")
        objectives = ["平衡收益与风险", "尽可能降低波动", "达到目标收益", "控制风险上限", "获得最高夏普比率"]
        objective = st.selectbox("配置目标", objectives, index=objectives.index(saved.get("objective", objectives[0])), key="objective")
        aversion = st.slider("风险厌恶系数 A", 0.5, 20.0, saved.get("aversion", 4.0), .5, key="aversion",
                             help="仅用于平衡收益与风险：最大化 E[r] − A×方差/2。A 越大，越偏好低波动。")
        target = st.number_input("目标年化收益 (%)", min_value=-50., max_value=100., value=saved.get("target", 6.), step=.5, key="target",
                                 help="仅在选择达到目标收益时使用；收益至少达到目标后，尽量降低方差。")
        cap = st.number_input("最高年化波动 (%)", min_value=0., max_value=100., value=saved.get("cap", 12.), step=.5, key="risk_cap",
                              help="仅在选择控制风险上限时使用；在该上限内追求最高预期收益。")
        budget = st.number_input("用于展示配置金额的本金（元）", min_value=1000., max_value=1e12, value=saved.get("budget", 100000.), step=10000., key="budget")
        st.caption("目标收益和波动上限影响推荐方案；机会集图仍显示全部满足持仓限制的组合。波动上限不等于最大亏损承诺。")
    with right, st.container(border=True):
        st.subheader("现金、做空与杠杆")
        include_rf = st.checkbox("允许配置无风险资产", value=saved.get("include_rf", True), key="include_rf")
        rf_rate = st.number_input("年化无风险利率 (%)", min_value=-5., max_value=30., value=saved.get("rf_rate", 2.), step=.1, key="rf_rate",
                        help="作为现金收益及夏普比率的参考利率；借款暂按同一利率。")
        rf_lower, rf_upper = 0., 100.
        if include_rf:
            a, b = st.columns(2)
            rf_lower = a.number_input("现金最低比例 (%)", min_value=-100., max_value=100., value=saved.get("rf_lower", 0.), step=5., key="rf_lower",
                                      help="负值表示允许按无风险利率借款。")
            rf_upper = b.number_input("现金最高比例 (%)", min_value=0., max_value=200., value=saved.get("rf_upper", 100.), step=5., key="rf_upper")
        allow_short = st.checkbox("允许风险资产做空", value=saved.get("allow_short", False), key="allow_short",
                                  help="开启后可在下表把单项最低比例设为负数；总敞口上限需同时高于 100%。")
        gross_limit = st.slider("总敞口上限 (%)", 100., 300., saved.get("gross_limit", 100.), 10., key="gross_limit",
                                help="所有资产（含现金）的权重绝对值之和。例如 120% 股票和 −20% 现金，总敞口为 140%。")
        st.caption("所有资产净权重合计为 100%。做空和借款均须同时满足总敞口上限。")
    st.subheader("每项资产的配置范围")
    st.caption("把最高比例设为 0% 可禁止投资；把最低和最高设为同一个值可固定持仓。")
    bound_data = pd.DataFrame({"资产": market.names, "最低比例 (%)": [-20. if allow_short else 0.]*len(market.names),
                               "最高比例 (%)": [100.]*len(market.names)})
    for i, name in enumerate(market.names):
        if name in saved.get("bounds", {}):
            lo, hi = saved["bounds"][name]
            bound_data.loc[i, "最低比例 (%)"] = lo if allow_short else max(0., lo)
            bound_data.loc[i, "最高比例 (%)"] = hi
    key = data_key+signature({"names": market.names, "short": allow_short})[:8]
    if st.session_state.get("page_changed") or "bound_base_"+key not in st.session_state:
        st.session_state["bound_base_"+key] = bound_data
    bound_values = st.data_editor(st.session_state["bound_base_"+key], hide_index=True, disabled=["资产"], key="bounds_editor_"+key,
        use_container_width=True, column_config={
            "最低比例 (%)": st.column_config.NumberColumn(min_value=-100. if allow_short else 0., max_value=100., step=5.),
            "最高比例 (%)": st.column_config.NumberColumn(min_value=0., max_value=200., step=5.)})
    use_groups = st.checkbox("增加资产类别限制（例如：权益最多 40%，债券至少 30%）", value=saved.get("use_groups", False), key="use_groups")
    group_bounds = {}
    if use_groups:
        group_data = pd.DataFrame({"类别": sorted(set(groups)), "最低比例 (%)": 0., "最高比例 (%)": 100.})
        for i, group in enumerate(group_data["类别"]):
            if group in saved.get("groups", {}):
                group_data.loc[i, ["最低比例 (%)", "最高比例 (%)"]] = saved["groups"][group]
        group_key = signature(groups)[:10]
        if st.session_state.get("page_changed") or "group_base_"+group_key not in st.session_state:
            st.session_state["group_base_"+group_key] = group_data
        group_values = st.data_editor(st.session_state["group_base_"+group_key], hide_index=True, disabled=["类别"],
            key="group_editor_"+group_key, use_container_width=True,
            column_config={"最低比例 (%)": st.column_config.NumberColumn(min_value=-100., max_value=100., step=5.),
                           "最高比例 (%)": st.column_config.NumberColumn(min_value=0., max_value=200., step=5.)})
        group_bounds = {r["类别"]: (r["最低比例 (%)"], r["最高比例 (%)"]) for _, r in group_values.iterrows()}
        st.caption("类别限制使用类别内资产的净权重之和；无风险资产单独受现金上下限约束。")
    st.session_state["settings"] = {
        "objective": objective, "aversion": aversion, "target": target, "cap": cap, "budget": budget,
        "include_rf": include_rf, "allow_short": allow_short, "rf_lower": rf_lower, "rf_upper": rf_upper, "rf_rate": rf_rate,
        "gross_limit": gross_limit, "use_groups": use_groups, "groups": group_bounds,
        "bounds": {r["资产"]: (r["最低比例 (%)"], r["最高比例 (%)"]) for _, r in bound_values.iterrows()},
    }
    try:
        market, groups, *_ = selected_market()
        full, rules, _, _ = settings(market, groups)
        opt = Optimizer(full, rules)
        low, high = opt.return_range()
        st.success(f"持仓限制可行 · 预期年化收益可达范围 {low.expected_return:.2%}–{high.expected_return:.2%}")
        st.button("生成我的配置方案 →", type="primary", on_click=navigate, args=(2,))
    except (AllocationError, TypeError, ValueError) as exc:
        st.error(str(exc))


@st.cache_data(show_spinner=False, max_entries=24)
def compute(names, mu, covariance, rf, rules, risky_rules, include_rf, objective, aversion, target, cap):
    risky = Market(names, mu, covariance, rf)
    full = risky.with_risk_free() if include_rf else risky
    opt = Optimizer(full, rules)
    frontier, efficient = opt.frontier()
    gmv = opt.minimum_variance()
    if objective == "尽可能降低波动":
        chosen = gmv
    elif objective == "达到目标收益":
        chosen = opt.target_return(target/100)
    elif objective == "控制风险上限":
        chosen = opt.under_risk_cap(cap/100)
    elif objective == "获得最高夏普比率":
        chosen = opt.maximum_sharpe()
    else:
        chosen = opt.utility(aversion)
    samples = opt.sample()
    tangent, tangent_error, risky_frontier, risky_efficient = None, None, [], []
    try:
        risky_opt = Optimizer(risky, risky_rules)
        if include_rf:
            risky_frontier, risky_efficient = risky_opt.frontier()
        tangent = risky_opt.maximum_sharpe()
    except AllocationError as exc:
        tangent_error = str(exc)
    return {"frontier": frontier, "efficient": efficient, "gmv": gmv, "chosen": chosen,
            "samples": samples, "tangent": tangent, "tangent_error": tangent_error,
            "risky_frontier": risky_frontier, "risky_efficient": risky_efficient}


def add_curve(fig, portfolios, name, color, dash=None, width=2):
    if portfolios:
        fig.add_trace(go.Scatter(x=[p.volatility*100 for p in portfolios], y=[p.expected_return*100 for p in portfolios],
                                mode="lines" if len(portfolios)>1 else "markers", name=name,
                                line={"color": color, "width": width, "dash": dash},
                                hovertemplate="年化波动 %{x:.2f}%<br>预期收益 %{y:.2f}%<extra>%{fullData.name}</extra>"))


def frontier_chart(result, market, opt, include_rf):
    fig = go.Figure()
    samples = result["samples"]
    fig.add_trace(go.Scattergl(x=[p.volatility*100 for p in samples], y=[p.expected_return*100 for p in samples],
        mode="markers", name="可行组合样本", marker={"color": "#B1C7D2", "size": 4, "opacity": .4},
        hovertemplate="年化波动 %{x:.2f}%<br>预期收益 %{y:.2f}%<extra>可行样本</extra>"))
    add_curve(fig, result["risky_frontier"], "仅风险资产 · 最小方差边界", "#ADB9C4", "dot")
    add_curve(fig, result["frontier"], "最小方差边界（完整）", "#98A6B4", "dash")
    add_curve(fig, result["efficient"], "有效前沿", "#147D72", width=4)
    tangent = result["tangent"]
    if include_rf and tangent is not None:
        extent = max(1.5, (opt.rules.gross_limit+1)/(np.abs(tangent.weights).sum()+1))
        scale = np.linspace(0, extent, 200)
        fig.add_trace(go.Scatter(x=scale*tangent.volatility*100,
            y=(market.rf+scale*(tangent.expected_return-market.rf))*100,
            mode="lines", name="资本配置线 CAL（理论参考）", line={"color": "#D8A44D", "dash": "dot"},
            hovertemplate="波动 %{x:.2f}%<br>收益 %{y:.2f}%<extra>需另检查投资限制</extra>"))
        interval = cal_interval(tangent.weights, opt.rules)
        if interval is not None:
            feasible_scale = np.asarray(interval)
            fig.add_trace(go.Scatter(x=feasible_scale*tangent.volatility*100,
                y=(market.rf+feasible_scale*(tangent.expected_return-market.rf))*100,
                mode="lines+markers" if abs(interval[1]-interval[0]) < 1e-8 else "lines",
                name="CAL 中满足当前限制的部分", line={"color": "#D8A44D", "width": 3}))
        fig.add_trace(go.Scatter(x=[tangent.volatility*100], y=[tangent.expected_return*100],
            mode="markers", name="风险资产最大夏普组合", marker={"color": "#D8A44D", "size": 11, "symbol": "diamond"}))
    for key, label, color, symbol in [("gmv", "全局最小方差组合", "#477EA8", "circle"),
                                      ("chosen", "你的配置方案", "#172C3C", "star")]:
        p = result[key]
        fig.add_trace(go.Scatter(x=[p.volatility*100], y=[p.expected_return*100], mode="markers", name=label,
            marker={"color": color, "size": 14, "symbol": symbol},
            hovertemplate="年化波动 %{x:.2f}%<br>预期收益 %{y:.2f}%<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(title="年化波动率 (%)", rangemode="tozero")
    fig.update_yaxes(title="预期年化收益 (%)")
    return chart_layout(fig, 540)


def weight_table(market, portfolio, budget):
    return pd.DataFrame({"资产": market.names, "配置比例 (%)": portfolio.weights*100,
                         "配置金额（元）": portfolio.weights*budget,
                         "预期收益贡献 (%)": portfolio.weights*market.mu*100})


def portfolio_rows(portfolios, names):
    rows = []
    for p in portfolios:
        rows.append({"expected_return": p.expected_return, "volatility": p.volatility,
                     "sharpe": p.sharpe, **dict(zip(names, p.weights))})
    return pd.DataFrame(rows)


def page_results():
    st.markdown('<div class="eyebrow">STEP 03 · YOUR ALLOCATION</div>', unsafe_allow_html=True)
    st.title("看见风险与收益的取舍。")
    try:
        market, groups, source, history, _ = selected_market()
        full, rules, risky_rules, s = settings(market, groups)
        opt = Optimizer(full, rules)
        with st.spinner("正在检查限制、计算前沿与配置方案…"):
            result = compute(market.names, market.mu, market.cov, market.rf, rules, risky_rules,
                             s.get("include_rf", True), s.get("objective", "平衡收益与风险"),
                             s.get("aversion", 4.), s.get("target", 6.), s.get("cap", 12.))
    except (AllocationError, ValueError, TypeError) as exc:
        st.error(str(exc))
        st.info("请回到第二步调整目标或投资限制，再生成配置方案。")
        st.button("返回调整投资要求", on_click=navigate, args=(1,))
        return
    chosen = result["chosen"]
    budget = s.get("budget", 100000.)
    include_rf = s.get("include_rf", True)
    objective = s.get("objective", "平衡收益与风险")
    st.markdown(f"**配置目标：{objective}**　·　{len(market.names)} 项风险资产　·　{'含无风险资产' if include_rf else '仅风险资产'}")
    a, b, c, d = st.columns(4)
    a.metric("预期年化收益", f"{chosen.expected_return:.2%}")
    b.metric("年化波动率", f"{chosen.volatility:.2%}")
    c.metric("夏普比率", "不适用" if chosen.sharpe is None else f"{chosen.sharpe:.2f}")
    d.metric("总敞口", f"{np.abs(chosen.weights).sum():.1%}")
    tabs = st.tabs(["我的方案", "投资机会集", "专业分析", "导出与复现"])
    with tabs[0]:
        left, right = st.columns([1, 1.4])
        with left:
            st.subheader("资金如何分配")
            weights = chosen.weights
            if np.any(weights < -1e-6):
                fig = go.Figure(go.Bar(x=weights*100, y=full.names, orientation="h",
                    marker_color=[COLORS[i % len(COLORS)] if w>=0 else "#AF7569" for i,w in enumerate(weights)],
                    hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
                fig.update_xaxes(title="占本金比例 (%)")
            else:
                visible = np.flatnonzero(weights > 1e-6)
                fig = go.Figure(go.Pie(labels=[full.names[i] for i in visible], values=weights[visible], hole=.68,
                    marker_colors=[COLORS[i % len(COLORS)] for i in visible], sort=False,
                    textinfo="label+percent", hovertemplate="%{label}: %{percent}<extra></extra>"))
                fig.update_layout(showlegend=False)
            st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
        with right:
            st.subheader("你的配置清单")
            table = weight_table(full, chosen, budget)
            st.dataframe(table.style.format({"配置比例 (%)": "{:.2f}", "配置金额（元）": "{:,.2f}",
                                            "预期收益贡献 (%)": "{:.2f}"}), hide_index=True, use_container_width=True)
            st.caption(f"本金 {budget:,.0f} 元。金额为理论配置，不含交易费用及最小交易单位；负金额表示做空或借款。")
            explanation = {"平衡收益与风险": f"按照 A = {s.get('aversion', 4.):g} 的风险厌恶系数，在所有持仓限制内最大化均值–方差效用。",
                "尽可能降低波动": "这是所有满足持仓限制的组合中，年化方差最低的组合。",
                "达到目标收益": f"在预期年化收益至少达到 {s.get('target', 6.):g}% 的组合中，选择方差最低的组合。",
                "控制风险上限": f"在年化波动不超过 {s.get('cap', 12.):g}% 的组合中，选择预期收益最高的组合。",
                "获得最高夏普比率": "在当前持仓限制内，最大化相对于无风险利率的预期超额收益与波动之比。"}
            st.info(explanation[objective])
        st.subheader("投资限制核对")
        checks = [{"限制": "净权重合计", "当前值": f"{chosen.weights.sum():.2%}", "要求": "100%", "状态": "通过"},
                  {"限制": "总敞口", "当前值": f"{np.abs(chosen.weights).sum():.2%}", "要求": f"≤ {rules.gross_limit:.0%}", "状态": "通过"}]
        for i, name in enumerate(full.names):
            checks.append({"限制": name, "当前值": f"{chosen.weights[i]:.2%}",
                           "要求": f"{rules.lower[i]:.0%} 至 {rules.upper[i]:.0%}", "状态": "通过"})
        for group, indices, lo, hi in rules.groups:
            checks.append({"限制": group+"类别", "当前值": f"{chosen.weights[list(indices)].sum():.2%}",
                           "要求": f"{lo:.0%} 至 {hi:.0%}", "状态": "通过"})
        st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)
        st.caption("所有通过状态均基于计算后的实际权重检查；求解容差为 0.0002 个百分点。")
    with tabs[1]:
        st.subheader("每一个点，都是一种配置可能")
        st.plotly_chart(frontier_chart(result, market, opt, include_rf), use_container_width=True)
        st.caption("灰色散点仅为可行组合的随机样本，不等于完整可行集；边界通过逐目标收益优化计算。有效前沿是全局最小方差点上方的部分。")
        if include_rf:
            st.info("加入无风险资产后的前沿已按全部限制重新优化。金色虚线是以风险资产最大夏普组合构造的理论 CAL，金色实线标出满足当前持仓限制的部分；有下限、上限或杠杆限制时，CAL 不一定与受约束前沿重合。")
        if result["tangent_error"]:
            st.warning("风险资产最大夏普组合："+result["tangent_error"])
        st.subheader("探索一项可行组合")
        pool = result["samples"]
        index = st.slider("选择样本编号", 1, max(2, len(pool)), 1, key="sample_index")
        sample = pool[min(index-1, len(pool)-1)]
        st.write(f"预期年收益 **{sample.expected_return:.2%}**　·　年化波动 **{sample.volatility:.2%}**")
        st.dataframe(weight_table(full, sample, budget).style.format({"配置比例 (%)": "{:.2f}",
            "配置金额（元）": "{:,.2f}", "预期收益贡献 (%)": "{:.2f}"}), hide_index=True, use_container_width=True)
    with tabs[2]:
        st.subheader("关键组合对比")
        comps = {"推荐方案": chosen, "全局最小方差": result["gmv"]}
        compare = [{"组合": label, "预期年收益 (%)": p.expected_return*100, "年化波动 (%)": p.volatility*100,
                    "夏普比率": p.sharpe} for label, p in comps.items()]
        if result["tangent"] is not None:
            p = result["tangent"]
            compare.append({"组合": "仅风险资产最大夏普", "预期年收益 (%)": p.expected_return*100,
                            "年化波动 (%)": p.volatility*100, "夏普比率": p.sharpe})
        st.dataframe(pd.DataFrame(compare).style.format(precision=3, na_rep="不适用"), hide_index=True, use_container_width=True)
        if result["tangent"] is not None:
            with st.expander("查看风险资产最大夏普组合的具体权重"):
                st.dataframe(weight_table(market, result["tangent"], budget).style.format(precision=3),
                             hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        vol = np.sqrt(np.diag(market.cov))
        corr = market.cov/np.outer(vol, vol)
        with a:
            st.subheader("相关系数")
            fig = go.Figure(go.Heatmap(z=corr, x=market.names, y=market.names,
                colorscale=[[0, "#AF7569"], [.5, "#F7F9FB"], [1, "#147D72"]], zmin=-1, zmax=1,
                text=np.round(corr, 2), texttemplate="%{text:.2f}", hovertemplate="%{x} × %{y}<br>相关系数 %{z:.3f}<extra></extra>"))
            st.plotly_chart(chart_layout(fig, 380), use_container_width=True)
        with b:
            st.subheader("风险资产年化参数")
            st.dataframe(pd.DataFrame({"资产": market.names, "预期收益 (%)": market.mu*100,
                "标准差 (%)": vol*100, "方差": np.diag(market.cov)}).style.format({"预期收益 (%)": "{:.2f}",
                    "标准差 (%)": "{:.2f}", "方差": "{:.6f}"}), hide_index=True, use_container_width=True)
        with st.expander("完整年化协方差矩阵（小数收益率的平方）"):
            st.dataframe(pd.DataFrame(full.cov, index=full.names, columns=full.names).style.format("{:.6f}"), use_container_width=True)
        st.subheader("风险贡献")
        if chosen.volatility > 1e-9:
            contribution = chosen.weights*(full.cov@chosen.weights)/chosen.volatility
            st.dataframe(pd.DataFrame({"资产": full.names, "对组合波动的贡献 (百分点)": contribution*100,
                "占组合风险 (%)": contribution/chosen.volatility*100}).style.format(precision=3), hide_index=True, use_container_width=True)
            st.caption("贡献之和等于组合波动。负贡献表示该资产对组合产生对冲作用。")
        else:
            st.info("当前组合波动为零，风险贡献比例与夏普比率不适用。")
    with tabs[3]:
        st.subheader("带走方案，也保留计算依据")
        st.caption("CSV 和 JSON 中的收益、波动与权重使用小数。例如 0.05 表示 5%。配置清单另以百分数显示。")
        st.download_button("下载配置清单 CSV", weight_table(full, chosen, budget).to_csv(index=False).encode("utf-8-sig"),
                           "allocation.csv", "text/csv")
        st.download_button("下载完整最小方差边界 CSV", portfolio_rows(result["frontier"], full.names).to_csv(index=False).encode("utf-8-sig"),
                           "minimum_variance_frontier.csv", "text/csv")
        st.download_button("下载有效前沿 CSV", portfolio_rows(result["efficient"], full.names).to_csv(index=False).encode("utf-8-sig"),
                           "efficient_frontier.csv", "text/csv")
        st.download_button("下载可行组合样本 CSV", portfolio_rows(result["samples"], full.names).to_csv(index=False).encode("utf-8-sig"),
                           "feasible_samples.csv", "text/csv")
        configuration = {"version": 1, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), "source": source,
            "names": market.names, "groups": groups, "annual_returns": market.mu.tolist(),
            "annual_covariance": market.cov.tolist(), "risk_free_rate": market.rf, "settings": s,
            "result": {"names": full.names, "weights": chosen.weights.tolist(), "expected_return": chosen.expected_return,
                       "volatility": chosen.volatility, "sharpe": chosen.sharpe}}
        json_text = json.dumps(configuration, ensure_ascii=False, indent=2, allow_nan=False)
        st.download_button("下载完整参数与结果 JSON", json_text.encode("utf-8"), "allocation_snapshot.json", "application/json")
        report = make_report(configuration, rules)
        st.download_button("下载课堂演示报告 Markdown", report.encode("utf-8"), "allocation_report.md", "text/markdown")
        st.caption("可用项目内的 reproduce.py 读取 JSON 并重新计算。历史上传数据只在当前服务进程中处理，本应用不主动写入文件或数据库。")
    st.markdown('<div class="foot">均值–方差模型基于你输入的预期参数；计算结果用于课程演示，不构成未来收益保证。这里的风险指标准差，不代表最大回撤或最大亏损。</div>', unsafe_allow_html=True)


def make_report(config, rules):
    result = config["result"]
    rows = ["# 资产配置演示报告", "", f"生成时间：{config['created_at']}", f"数据来源：{config['source']}",
            f"配置目标：{config['settings'].get('objective', '平衡收益与风险')}", "",
            f"预期年化收益：{result['expected_return']:.2%}", f"年化波动率：{result['volatility']:.2%}",
            f"夏普比率：{result['sharpe'] if result['sharpe'] is not None else '不适用'}", "",
            "| 资产 | 权重 | 最低比例 | 最高比例 |", "| --- | ---: | ---: | ---: |"]
    rows += [f"| {name} | {w:.2%} | {lo:.2%} | {hi:.2%} |" for name, w, lo, hi in
             zip(result["names"], result["weights"], rules.lower, rules.upper)]
    rows += ["", f"总敞口上限：{rules.gross_limit:.0%}", ""]
    for group, _, lo, hi in rules.groups:
        rows.append(f"类别限制：{group} {lo:.0%} 至 {hi:.0%}")
    rows += ["", "使用算术均值、样本协方差或用户指定的年化参数，按净权重合计 100% 优化。",
             "示例资产与历史序列为人工模拟数据；导入数据的估计参数以用户文件为准。",
             "完整输入和结果见同次导出的 JSON，可用 reproduce.py 复现。",
             "风险衡量为年化标准差，收益目标为预期值，不保证实际收益。"]
    return "\n".join(rows)


def page_methods():
    st.markdown('<div class="eyebrow">METHODOLOGY · COURSE CHECKLIST</div>', unsafe_allow_html=True)
    st.title("从课堂理论，到可使用的工具。")
    st.write("本工具采用单期均值–方差框架。所有输入统一年化；投资者选择资产与限制后，工具重新计算可行组合与前沿。")
    st.subheader("模型与计算")
    st.latex(r"\mathbb{E}[r_p]=w^\top\mu,\qquad \sigma_p=\sqrt{w^\top\Sigma w},\qquad \mathbf{1}^\top w=1")
    st.latex(r"\min_w w^\top\Sigma w\quad \mathrm{s.t.}\quad w^\top\mu=r^*,\quad l\leq w\leq u,\quad \sum_i |w_i|\leq L")
    st.write("类别比例是组内净权重之和，可另加上下限。先用线性规划检查持仓限制是否可行，再逐目标收益求最小方差；有效前沿取全局最小方差组合以上的部分。")
    st.write("导入简单收益率后，预期年收益采用每期样本均值 × 年内期数，年化协方差采用样本协方差 × 年内期数。这是假设收益平稳、跨期不相关的算术年化，不是复合年增长率。")
    st.latex(r"U(w)=w^\top\mu-\frac{A}{2}w^\top\Sigma w,\qquad S(w)=\frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}}")
    st.write("最大夏普组合在存在正超额收益时，用凸透视变换求解：令 y=t·w，固定超额收益为 1，最小化 y 的方差，再还原权重。仅风险资产的该组合在标准条件下是切点组合；约束边界上也可能是角点。")
    st.subheader("加入无风险资产时")
    st.latex(r"\mu_c=r_f+y(\mu_T-r_f),\qquad \sigma_c=|y|\sigma_T")
    st.write("无风险资产方差及与风险资产的协方差为零。工具直接扩展协方差矩阵并重新优化，不把整条理论 CAL 当成可投资前沿。有最低持仓、比例上限、现金限制或总敞口限制时，风险组合缩放后可能违反限制；图中单独检查并标出可行的 CAL 部分。")
    st.write("若允许全部配置无风险资产，全局最小方差组合通常是 100% 无风险资产。零波动时不显示夏普比率。若没有正超额收益的可行组合，则不报告有限的正超额收益切点组合。")
    st.subheader("附录功能逐项对照")
    checklist = [
        ("投资者决定必要输入", "第一步参数表、第二步投资偏好与限制"),
        ("识别所有可投资资产", "启用资产清单、可新增资产、CSV 导入"),
        ("预期收益、方差或标准差", "可编辑的年化假设；历史收益样本估计；专业分析参数表"),
        ("所有资产对的相关性或协方差", "相关矩阵编辑、相关性热力图、完整协方差矩阵"),
        ("识别各资产投资限制", "单项上下限、禁止投资、类别限制、现金、做空和总敞口"),
        ("可行集及每个组合的权重、收益与风险", "随机可行样本图、样本探索、全部样本 CSV；边界精确优化"),
        ("风险资产的最小方差边界、全局最小方差、有效前沿", "取消无风险资产开关；查看机会集和专业分析"),
        ("加入无风险资产后的三类输出", "启用无风险资产；对扩展资产集合重新优化"),
        ("切点／最优风险组合与资本配置线", "风险资产最大夏普组合、具体权重、CAL 理论线及满足限制部分"),
    ]
    st.dataframe(pd.DataFrame(checklist, columns=["作业要求", "本工具实现"]), hide_index=True, use_container_width=True)
    st.subheader("演示时可以尝试")
    st.write("1. 保留默认数据，比较有现金和没有现金时的全局最小方差组合。\n2. 把权益类别上限调到 40%，观察前沿与权重变化。\n3. 将每项资产最高比例设为 10%，查看工具如何解释不可行。\n4. 允许借款和更高总敞口，观察 CAL 可行部分如何变化。\n5. 导入示例 CSV，检查样本估计与手工假设的区别。")
    st.caption("模型暂不考虑税费、整数手数、交易成本、再平衡、汇率预测与多期现金流。做空和借款使用统一无风险参考利率，不模拟保证金追缴。")


style()
with st.sidebar:
    st.markdown("### ◒ 配比")
    st.caption("PORTFOLIO LAB / 资产配置工具箱")
    st.divider()
    page = st.radio("你的配置流程", STEPS, key="navigation", label_visibility="collapsed")
    st.divider()
    st.markdown("**把投资想法，变成可检验的配置。**")
    st.caption("Python · 均值–方差优化\n\n课程 Demo · 无需注册")
    st.markdown('<div class="foot">先选择资产，再定义规则。<br>工具负责计算，你决定取舍。</div>', unsafe_allow_html=True)

st.session_state["page_changed"] = page != st.session_state.get("last_page")
st.session_state["last_page"] = page
if page == STEPS[0]:
    page_assets()
elif page == STEPS[1]:
    page_settings()
elif page == STEPS[2]:
    page_results()
else:
    page_methods()
