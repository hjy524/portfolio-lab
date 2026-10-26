import unittest
import numpy as np

from portfolio.core import AllocationError, Market, Optimizer, Restrictions
from portfolio.data import demo_market, estimate_from_returns


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.market = Market(["A", "B"], [.10, .20], np.diag([.01, .04]), .02)
        self.opt = Optimizer(self.market, Restrictions(np.zeros(2), np.ones(2)))

    def test_analytic_global_minimum(self):
        p = self.opt.minimum_variance()
        np.testing.assert_allclose(p.weights, [.8, .2], atol=1e-6)
        self.assertAlmostEqual(p.volatility, np.sqrt(.008), places=7)

    def test_analytic_tangency(self):
        p = self.opt.maximum_sharpe()
        np.testing.assert_allclose(p.weights, [.64, .36], atol=2e-5)

    def test_exact_target(self):
        p = self.opt.minimum_variance(.15)
        np.testing.assert_allclose(p.weights, [.5, .5], atol=1e-6)
        self.assertAlmostEqual(p.expected_return, .15)

    def test_frontier_lower_branch(self):
        frontier, efficient = self.opt.frontier(15)
        self.assertAlmostEqual(frontier[0].expected_return, .1)
        self.assertTrue(any(p.expected_return < .12-1e-6 for p in frontier))
        self.assertTrue(all(p.expected_return >= .12-1e-6 for p in efficient))

    def test_utility_analytic(self):
        # derivative of mu'w - (4/2) w'Cov w with w2=1-w1.
        p = self.opt.utility(4)
        np.testing.assert_allclose(p.weights, [.3, .7], atol=1e-6)

    def test_target_below_gmv_uses_gmv(self):
        np.testing.assert_allclose(self.opt.target_return(.11).weights, [.8, .2], atol=1e-6)

    def test_target_unreachable(self):
        with self.assertRaisesRegex(AllocationError, "不可达"):
            self.opt.target_return(.3)

    def test_risk_cap_unreachable(self):
        with self.assertRaisesRegex(AllocationError, "不可达"):
            self.opt.under_risk_cap(.01)

    def test_risk_cap_boundary(self):
        p = self.opt.under_risk_cap(.12)
        self.assertLessEqual(p.volatility, .120002)
        self.assertGreater(p.expected_return, .15)

    def test_zero_risk_is_exact(self):
        m = self.market.with_risk_free()
        opt = Optimizer(m, Restrictions(np.zeros(3), np.ones(3)))
        p = opt.minimum_variance()
        np.testing.assert_array_equal(p.weights, [0, 0, 1])
        self.assertEqual(p.volatility, 0)
        self.assertIsNone(p.sharpe)
        frontier, _ = opt.frontier(10)
        self.assertEqual(frontier[0].volatility, 0)

    def test_shorting_gross_leverage(self):
        opt = Optimizer(self.market, Restrictions(np.array([-.5, -.5]), np.array([1.5, 1.5]), 1.4))
        p = opt.utility(.5)
        self.assertTrue(opt.is_feasible(p.weights))
        np.testing.assert_allclose(p.weights, [-.2, 1.2], atol=2e-6)
        for sample in opt.sample(40):
            self.assertTrue(opt.is_feasible(sample.weights))

    def test_borrowing(self):
        m = self.market.with_risk_free()
        opt = Optimizer(m, Restrictions(np.array([0., 0., -.5]), np.array([1.5, 1.5, 1.]), 1.6))
        p = opt.utility(.5)
        self.assertTrue(opt.is_feasible(p.weights))
        self.assertGreater(np.abs(p.weights).sum(), 1)
        self.assertLess(p.weights[-1], 0)
        opt.frontier(12)

    def test_group_limit(self):
        opt = Optimizer(self.market, Restrictions(np.zeros(2), np.ones(2), 1., (("A group", (0,), .2, .4),)))
        p = opt.minimum_variance()
        self.assertAlmostEqual(p.weights[0], .4, places=5)
        opt.frontier(12)

    def test_fixed_weights(self):
        opt = Optimizer(self.market, Restrictions(np.array([.4, .6]), np.array([.4, .6])))
        p = opt.minimum_variance()
        np.testing.assert_allclose(p.weights, [.4, .6])
        self.assertEqual(len(opt.frontier()[0]), 1)
        self.assertEqual(len(opt.sample(20)), 1)

    def test_one_asset_forbidden(self):
        opt = Optimizer(self.market, Restrictions(np.zeros(2), np.array([0., 1.])))
        np.testing.assert_allclose(opt.minimum_variance().weights, [0, 1])
        np.testing.assert_allclose(opt.maximum_sharpe().weights, [0, 1], atol=1e-6)

    def test_impossible_bounds(self):
        with self.assertRaisesRegex(AllocationError, "不足"):
            Optimizer(self.market, Restrictions(np.zeros(2), np.array([.1, .1])))

    def test_group_conflict(self):
        with self.assertRaisesRegex(AllocationError, "冲突"):
            Optimizer(self.market, Restrictions(np.zeros(2), np.ones(2), 1.,
                (("A", (0,), .7, 1.), ("B", (1,), .7, 1.))))

    def test_non_psd_rejected(self):
        with self.assertRaisesRegex(AllocationError, "半正定"):
            Market(["A", "B"], [.1, .2], [[1., 2.], [2., 1.]])

    def test_no_positive_excess(self):
        m = Market(["A", "B"], [.01, .015], np.diag([.01, .04]), .02)
        with self.assertRaisesRegex(AllocationError, "正超额"):
            Optimizer(m, Restrictions(np.zeros(2), np.ones(2))).maximum_sharpe()

    def test_equal_returns(self):
        m = Market(["A", "B"], [.1, .1], np.diag([.01, .04]))
        opt = Optimizer(m, Restrictions(np.zeros(2), np.ones(2)))
        self.assertEqual(len(opt.frontier()[0]), 1)
        np.testing.assert_allclose(opt.minimum_variance(.1).weights, [.8, .2], atol=1e-6)

    def test_samples_respect_fixed_category(self):
        a, corr, _ = demo_market()
        m = Market(a.name, a["return"], corr*np.outer(a.volatility, a.volatility))
        opt = Optimizer(m, Restrictions(np.zeros(6), np.ones(6), 1., (("equity", (0, 1), .4, .4),)))
        samples = opt.sample(80)
        self.assertGreater(len(samples), 10)
        self.assertTrue(all(opt.is_feasible(p.weights) for p in samples))


class ReturnImportTests(unittest.TestCase):
    def setUp(self):
        self.assets, self.corr, self.history = demo_market()
        self.frame = self.history.reset_index()

    def test_sample_estimates_match_demo(self):
        assets, corr, _ = estimate_from_returns(self.frame, 12)
        np.testing.assert_allclose(assets["return"], self.assets["return"], atol=1e-10)
        np.testing.assert_allclose(assets.volatility, self.assets.volatility, atol=1e-10)
        np.testing.assert_allclose(corr, self.corr, atol=1e-10)

    def test_missing_observation_rejected(self):
        self.frame.iloc[2, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "缺失"):
            estimate_from_returns(self.frame, 12)

    def test_duplicate_date_rejected(self):
        self.frame.loc[1, "date"] = self.frame.loc[0, "date"]
        with self.assertRaisesRegex(ValueError, "重复"):
            estimate_from_returns(self.frame, 12)

    def test_duplicate_asset_rejected(self):
        self.frame.iloc[:, 2] = self.frame.iloc[:, 1]
        with self.assertRaisesRegex(ValueError, "奇异"):
            estimate_from_returns(self.frame, 12)


if __name__ == "__main__":
    unittest.main()
