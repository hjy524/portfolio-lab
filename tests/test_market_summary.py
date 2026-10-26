import copy
from datetime import date
import unittest

import numpy as np
import pandas as pd

from portfolio.core import Market
from portfolio.guidance import beginner_settings
from portfolio.market_data import load_market_snapshot, parse_eastmoney, parse_yahoo, ETF_UNIVERSE
from portfolio.summary import comparison_table, weights_comparison, one_line, largest_holding


class MarketDataTests(unittest.TestCase):
    def test_verified_snapshot_is_real_aligned_and_identified(self):
        assets, corr, history, meta = load_market_snapshot()
        self.assertEqual(assets.code.tolist(), [a["code"] for a in ETF_UNIVERSE])
        self.assertGreaterEqual(len(history), 252)
        self.assertEqual(meta["currency"], "CNY")
        self.assertTrue(meta["retrieved_at"].endswith("+08:00"))
        self.assertTrue(np.isfinite(history.to_numpy()).all())
        self.assertGreater(np.linalg.eigvalsh(corr).min(), 0)
        self.assertIn("公开历史行情", meta["provider"])

    def yahoo(self):
        dates = pd.bdate_range("2020-01-01", periods=270)
        return {"chart": {"error": None, "result": [{"meta": {"symbol": "510300.SS", "currency": "CNY"},
            "timestamp": [int(x.tz_localize("Asia/Shanghai").timestamp()) for x in dates],
            "indicators": {"adjclose": [{"adjclose": list(np.linspace(3., 4., len(dates)))}]}}]}}

    def test_yahoo_adjusted_closes_and_date_filter(self):
        s = parse_yahoo(self.yahoo(), "510300", date(2020, 1, 1), date(2021, 12, 31))
        self.assertEqual(len(s), 270)
        self.assertEqual(s.name, "510300")
        self.assertAlmostEqual(s.iloc[-1], 4.)

    def test_wrong_currency_is_rejected(self):
        data = self.yahoo()
        data["chart"]["result"][0]["meta"]["currency"] = "USD"
        with self.assertRaisesRegex(ValueError, "币种"):
            parse_yahoo(data, "510300", date(2020, 1, 1), date(2021, 12, 31))

    def test_unadjusted_substitution_is_rejected(self):
        data = self.yahoo()
        data["chart"]["result"][0]["indicators"] = {"quote": [{"close": [4.]*270}]}
        with self.assertRaisesRegex(ValueError, "未复权"):
            parse_yahoo(data, "510300", date(2020, 1, 1), date(2021, 12, 31))

    def test_wrong_eastmoney_security_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "验证"):
            parse_eastmoney({"data": {"code": "510500", "klines": ["2020-01-01,1,1"]}}, "510300", date(2020, 1, 1), date(2021, 12, 31))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.market = Market(["股票", "债券"], [.1, .04], np.diag([.04, .0025]))
        self.metadata = {"股票": {"code": "510300", "group": "权益"}, "债券": {"code": "511010", "group": "债券"},
                         "无风险资产": {"code": "—（模型基准）", "group": "现金"}}

    def test_summary_contains_top_security_and_risk(self):
        p = self.market.portfolio([.3, .7])
        frame = comparison_table([("我的方案", self.market.names, p)], self.metadata)
        self.assertEqual(frame.iloc[0]["证券／资产代码"], "511010")
        self.assertAlmostEqual(frame.iloc[0]["最大仓位 (%)"], 70.)
        self.assertIn("511010", one_line(self.market.names, p, self.metadata))

    def test_short_exposure_is_counted_and_named(self):
        p = self.market.portfolio([-1.5, 2.5])
        self.assertEqual(largest_holding(self.market.names, p, self.metadata)["name"], "债券")

    def test_cash_does_not_get_fake_stock_code(self):
        m = self.market.with_risk_free()
        p = m.portfolio([0., 0., 1.])
        self.assertIsNone(p.sharpe)
        self.assertIn("模型基准", largest_holding(m.names, p, self.metadata)["code"])
        self.assertIn("波动为零", one_line(m.names, p, self.metadata))

    def test_category_and_asset_weights_sum_correctly(self):
        m = self.market.with_risk_free()
        plans = [("我的方案", m.names, m.portfolio([.2, .3, .5])),
                 ("切线组合", self.market.names, self.market.portfolio([.6, .4]))]
        for categories in [True, False]:
            table = weights_comparison(plans, self.metadata, categories)
            self.assertAlmostEqual(table["我的方案 (%)"].sum(), 100)
            self.assertAlmostEqual(table["切线组合 (%)"].sum(), 100)

    def test_beginner_horizon_and_reserve_affect_actual_rules(self):
        s = beginner_settings(["股票", "债券"], ["权益", "债券"], "愿意接受更大起伏", "半年内可能用到", 10., 100000.)
        self.assertEqual(s["rf_lower"], 50.)
        self.assertEqual(s["groups"]["权益"][1], 10.)
        self.assertFalse(s["allow_short"])
        self.assertEqual(s["gross_limit"], 100.)
        self.assertEqual(s["bounds"]["股票"], [0., 40.])


if __name__ == "__main__":
    unittest.main()
