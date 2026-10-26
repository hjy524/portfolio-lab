"""Constrained mean–variance optimization; no UI or external data dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import OptimizeWarning, linprog, minimize

TOL = 2e-6


class AllocationError(ValueError):
    """An input, feasibility, or numerical issue that can be shown to investors."""


@dataclass
class Portfolio:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float | None


class Market:
    def __init__(self, names, returns, covariance, rf=0.02):
        self.names = list(names)
        self.mu = np.asarray(returns, dtype=float)
        self.cov = np.asarray(covariance, dtype=float)
        self.rf = float(rf)
        n = len(self.names)
        if not n or self.mu.shape != (n,) or self.cov.shape != (n, n):
            raise AllocationError("资产名称、收益与协方差矩阵的维度不一致。")
        if len(set(self.names)) != n:
            raise AllocationError("资产名称必须唯一。")
        if not np.isfinite(self.mu).all() or not np.isfinite(self.cov).all() or not np.isfinite(self.rf):
            raise AllocationError("收益、协方差与无风险利率必须为有限数字。")
        if not np.allclose(self.cov, self.cov.T, atol=1e-10, rtol=0):
            raise AllocationError("相关系数或协方差矩阵必须对称。")
        if np.linalg.eigvalsh(self.cov).min() < -1e-10:
            raise AllocationError("相关系数矩阵不是半正定矩阵，请修正资产之间的相关关系。")
        if np.any(np.diag(self.cov) < 0):
            raise AllocationError("资产方差不能为负。")

    def portfolio(self, weights):
        weights = np.asarray(weights, dtype=float)
        mean = float(weights @ self.mu)
        vol = float(np.sqrt(max(0, weights @ self.cov @ weights)))
        return Portfolio(weights.copy(), mean, vol, (mean - self.rf) / vol if vol > 1e-9 else None)

    def with_risk_free(self):
        n = len(self.mu)
        cov = np.zeros((n + 1, n + 1))
        cov[:n, :n] = self.cov
        return Market(self.names + ["无风险资产"], np.r_[self.mu, self.rf], cov, self.rf)


@dataclass
class Restrictions:
    lower: np.ndarray
    upper: np.ndarray
    gross_limit: float = 1.0
    # Each rule: (display label, integer asset indices, lower net weight, upper net weight).
    groups: tuple = ()


class Optimizer:
    """Use LP for exact linear feasibility and SLSQP for convex quadratic problems.

    Auxiliary u variables enforce |w_i| <= u_i and sum u_i <= gross_limit.
    This keeps all restrictions linear even when short positions are permitted.
    """
    def __init__(self, market: Market, restrictions: Restrictions):
        self.market = market
        self.rules = restrictions
        self.n = len(market.mu)
        n = self.n
        lo, hi = np.asarray(restrictions.lower, float), np.asarray(restrictions.upper, float)
        if lo.shape != (n,) or hi.shape != (n,) or not np.isfinite(np.r_[lo, hi]).all():
            raise AllocationError("请为每项资产填写有限的配置上下限。")
        if np.any(lo > hi):
            i = int(np.flatnonzero(lo > hi)[0])
            raise AllocationError(f"{market.names[i]} 的最低比例高于最高比例。")
        if not np.isfinite(restrictions.gross_limit) or restrictions.gross_limit < 1:
            raise AllocationError("总敞口上限不能低于 100%。")
        if lo.sum() > 1 + TOL:
            raise AllocationError(f"所有最低比例相加为 {lo.sum():.1%}，超过可用的 100%。")
        if hi.sum() < 1 - TOL:
            raise AllocationError(f"所有最高比例相加仅 {hi.sum():.1%}，不足以配置全部资金。")
        self.aux = bool(np.any(lo < 0))
        self.d = 2*n if self.aux else n
        self.bounds = list(zip(lo, hi)) + ([(0, restrictions.gross_limit)]*n if self.aux else [])
        self.aeq = np.zeros((1, self.d))
        self.aeq[0, :n] = 1
        self.beq = np.ones(1)
        rows, rhs = [], []
        for label, indices, lower, upper in restrictions.groups:
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise AllocationError(f"{label} 的类别比例上下限无效。")
            row = np.zeros(self.d)
            row[list(indices)] = 1
            rows.extend([row, -row])
            rhs.extend([upper, -lower])
        if self.aux:
            for i in range(n):
                row = np.zeros(self.d)
                row[i], row[n+i] = 1, -1
                rows.append(row)
                rhs.append(0)
                row = row.copy()
                row[i] = -1
                rows.append(row)
                rhs.append(0)
            row = np.r_[np.zeros(n), np.ones(n)]
            rows.append(row)
            rhs.append(restrictions.gross_limit)
        self.aub = np.asarray(rows) if rows else np.empty((0, self.d))
        self.bub = np.asarray(rhs)
        result = self._lp(np.zeros(n))
        if not result.success:
            raise AllocationError("投资限制之间存在冲突，无法分配全部资金。请放宽单项上下限、资产类别限制或总敞口上限。")
        self.start = result.x

    def _lp(self, c, extra_eq=None):
        objective = np.r_[c, np.zeros(self.d - self.n)]
        aeq, beq = self.aeq, self.beq
        if extra_eq is not None:
            row = np.r_[self.market.mu, np.zeros(self.d - self.n)]
            aeq, beq = np.vstack([aeq, row]), np.r_[beq, extra_eq]
        # Streamlit reruns execute on different threads. A single HiGHS worker
        # avoids its global parallel scheduler hanging across script runners.
        # SciPy 1.14 forwards this native HiGHS option, with a wrapper warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning,
                                    message="Unrecognized options detected:.*threads.*")
            return linprog(objective, A_ub=self.aub if len(self.aub) else None,
                           b_ub=self.bub if len(self.bub) else None,
                           A_eq=aeq, b_eq=beq, bounds=self.bounds, method="highs",
                           options={"threads": 1})

    def is_feasible(self, weights, tolerance=TOL):
        w = np.asarray(weights)
        if w.shape != (self.n,) or not np.isfinite(w).all():
            return False
        lo, hi = np.asarray(self.rules.lower), np.asarray(self.rules.upper)
        if abs(w.sum() - 1) > tolerance or np.any(w < lo-tolerance) or np.any(w > hi+tolerance):
            return False
        if np.abs(w).sum() > self.rules.gross_limit + tolerance:
            return False
        return all(lower-tolerance <= w[list(idx)].sum() <= upper+tolerance
                   for _, idx, lower, upper in self.rules.groups)

    def return_range(self):
        low, high = self._lp(self.market.mu), self._lp(-self.market.mu)
        return self.market.portfolio(low.x[:self.n]), self.market.portfolio(high.x[:self.n])

    def _solve(self, fun, jac, target=None, risk_cap=None, start=None):
        cons = [{"type": "eq", "fun": lambda x: self.aeq @ x-self.beq,
                 "jac": lambda x: self.aeq}]
        if len(self.aub):
            cons.append({"type": "ineq", "fun": lambda x: self.bub-self.aub@x,
                         "jac": lambda x: -self.aub})
        mu = np.r_[self.market.mu, np.zeros(self.d-self.n)]
        if target is not None:
            cons.append({"type": "eq", "fun": lambda x: x@mu-target, "jac": lambda x: mu})
        if risk_cap is not None:
            cons.append({"type": "ineq", "fun": lambda x: 100*(risk_cap**2-x[:self.n]@self.market.cov@x[:self.n]),
                         "jac": lambda x: np.r_[-200*self.market.cov@x[:self.n], np.zeros(self.d-self.n)]})
        result = minimize(fun, self.start if start is None else start, jac=jac,
                          bounds=self.bounds, constraints=cons, method="SLSQP",
                          options={"ftol": 1e-11, "maxiter": 700})
        w = result.x[:self.n]
        if not result.success or not self.is_feasible(w):
            raise AllocationError("本次优化未可靠收敛。请检查参数、放宽过紧的限制或减少高度相似的资产。")
        p = self.market.portfolio(w)
        if target is not None and abs(p.expected_return-target) > TOL:
            raise AllocationError("目标收益约束未达到精度要求。")
        if risk_cap is not None and p.volatility > risk_cap+TOL:
            raise AllocationError("风险上限未达到精度要求。")
        return p

    def minimum_variance(self, target=None):
        start = self.start
        # Return an exactly zero-risk portfolio when possible, rather than a
        # numerically tiny risky holding with a misleading finite Sharpe ratio.
        for i in np.flatnonzero(np.diag(self.market.cov) == 0):
            cash = np.eye(self.n)[i]
            if self.is_feasible(cash) and (target is None or abs(self.market.mu[i]-target) < 1e-12):
                return self.market.portfolio(cash)
        if target is not None:
            result = self._lp(np.zeros(self.n), extra_eq=target)
            if not result.success:
                low, high = self.return_range()
                raise AllocationError(f"目标收益不可达；当前可达年化收益范围为 {low.expected_return:.2%}–{high.expected_return:.2%}。")
            start = result.x
            # A constant return vector makes the second equality redundant.
            if np.ptp(self.market.mu) < 1e-12:
                target = None
        cov = self.market.cov
        return self._solve(lambda x: 100*x[:self.n]@cov@x[:self.n],
                           lambda x: np.r_[200*cov@x[:self.n], np.zeros(self.d-self.n)],
                           target=target, start=start)

    def utility(self, aversion):
        if not np.isfinite(aversion) or aversion <= 0:
            raise AllocationError("风险厌恶系数必须大于 0。")
        mu, cov = self.market.mu, self.market.cov
        return self._solve(lambda x: 100*(.5*aversion*x[:self.n]@cov@x[:self.n]-x[:self.n]@mu),
                           lambda x: np.r_[100*(aversion*cov@x[:self.n]-mu), np.zeros(self.d-self.n)])

    def target_return(self, target):
        gmv = self.minimum_variance()
        return gmv if gmv.expected_return >= target-TOL else self.minimum_variance(target)

    def under_risk_cap(self, cap):
        if not np.isfinite(cap) or cap < 0:
            raise AllocationError("风险上限必须是非负数。")
        gmv = self.minimum_variance()
        if gmv.volatility > cap + TOL:
            raise AllocationError(f"风险上限不可达；当前限制下最低年化波动为 {gmv.volatility:.2%}。")
        if abs(gmv.volatility-cap) < 1e-9:
            return gmv
        start = np.r_[gmv.weights, np.abs(gmv.weights)] if self.aux else gmv.weights
        return self._solve(lambda x: -100*x[:self.n]@self.market.mu,
                           lambda x: np.r_[-100*self.market.mu, np.zeros(self.d-self.n)],
                           risk_cap=cap, start=start)

    def maximum_sharpe(self):
        """Positive-excess Sharpe via a convex perspective transformation.

        Set y=t*w, (mu-rf)'y=1. Minimize y'Cov*y with scaled linear
        restrictions and t>=0. Recover w=y/t. No local ratio maximization.
        """
        excess = self.market.mu-self.market.rf
        _, high = self.return_range()
        if high.expected_return <= self.market.rf + 1e-10:
            raise AllocationError("所有可行组合的预期收益均不高于无风险利率，无法给出具有正超额收益的切点组合。")
        seed = self._lp(-self.market.mu).x
        t = 1 / (excess@seed[:self.n])
        start = np.r_[seed*t, t]
        size = self.d+1
        eq = np.zeros((2, size))
        eq[0, :self.d], eq[0, -1] = self.aeq[0], -1
        eq[1, :self.n] = excess
        rows = [np.r_[row, -bound] for row, bound in zip(self.aub, self.bub)]
        for i, (lower, upper) in enumerate(self.bounds):
            row = np.zeros(size)
            row[i], row[-1] = 1, -upper
            rows.append(row)
            row = np.zeros(size)
            row[i], row[-1] = -1, lower
            rows.append(row)
        aub = np.asarray(rows)
        cov = self.market.cov
        # Scale by seed variance to avoid tiny/huge transformed objectives.
        scale = 1/max(float(start[:self.n]@cov@start[:self.n]), 1e-8)
        result = minimize(lambda x: scale*x[:self.n]@cov@x[:self.n], start,
                          jac=lambda x: np.r_[2*scale*cov@x[:self.n], np.zeros(size-self.n)],
                          bounds=[(None, None)]*self.d + [(1e-9, None)],
                          constraints=[{"type": "eq", "fun": lambda x: eq@x-np.array([0, 1]), "jac": lambda x: eq},
                                       {"type": "ineq", "fun": lambda x: -aub@x, "jac": lambda x: -aub}],
                          method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        if not result.success or result.x[-1] <= 0:
            raise AllocationError("最大夏普组合未可靠收敛，请检查相关矩阵或放宽限制。")
        weights = result.x[:self.n]/result.x[-1]
        if not self.is_feasible(weights):
            raise AllocationError("最大夏普组合未通过投资限制检查。")
        p = self.market.portfolio(weights)
        if p.volatility < 1e-9:
            raise AllocationError("存在零风险且高于无风险利率的组合，有限的切点组合不存在。")
        return p

    def frontier(self, points=55):
        low, high = self.return_range()
        gmv = self.minimum_variance()
        if high.expected_return-low.expected_return < 1e-10:
            return [gmv], [gmv]
        targets = np.unique(np.r_[np.linspace(low.expected_return, high.expected_return, points), gmv.expected_return])
        portfolios = [self.minimum_variance(float(target)) for target in targets]
        efficient = [p for p in portfolios if p.expected_return >= gmv.expected_return-1e-8]
        return portfolios, efficient

    def sample(self, count=700, seed=42):
        """Hit-and-run sampling in the feasible polytope; display as a sample only."""
        rng = np.random.default_rng(seed)
        if self.aux:
            aub, bub = self.aub, self.bub
            eq = self.aeq
            x = self.start.copy()
        else:
            aub, bub, eq, x = self.aub, self.bub, self.aeq, self.start.copy()
        # Average feasible extreme points to avoid starting at a polytope corner.
        starts = [x]
        for _ in range(12):
            result = self._lp(rng.normal(size=self.n))
            if result.success:
                starts.append(result.x)
        x = np.mean(starts, axis=0)
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        # Include tight inequalities in the affine hull (fixed weights/group rules).
        all_a = np.vstack([aub, np.eye(self.d), -np.eye(self.d)])
        all_b = np.r_[bub, hi, -lo]
        tight = np.max(np.abs(np.asarray(starts)@all_a.T-all_b), axis=0) < 1e-9
        hull = np.vstack([eq, all_a[tight]])
        _, s, vh = np.linalg.svd(hull, full_matrices=True)
        rank = int((s > 1e-9).sum())
        basis = vh[rank:].T
        if basis.shape[1] == 0:
            return [self.market.portfolio(x[:self.n])]
        samples = []
        for step in range(count*4+120):
            direction = basis@rng.normal(size=basis.shape[1])
            direction /= max(np.linalg.norm(direction), 1e-12)
            denom, slack = all_a@direction, all_b-all_a@x
            positive, negative = denom > 1e-11, denom < -1e-11
            upper = np.min(slack[positive]/denom[positive]) if positive.any() else np.inf
            lower = np.max(slack[negative]/denom[negative]) if negative.any() else -np.inf
            if not np.isfinite([lower, upper]).all() or upper < lower:
                continue
            x = x + rng.uniform(lower, upper)*direction
            if step >= 120 and step % 4 == 0 and self.is_feasible(x[:self.n]):
                samples.append(self.market.portfolio(x[:self.n]))
            if len(samples) >= count:
                break
        return samples or [self.minimum_variance()]
