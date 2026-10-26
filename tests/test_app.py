"""Interaction tests use Streamlit's actual script runner; no paid services."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["01  选择资产", "02  设置目标与限制", "03  查看配置方案", "04  方法与作业对照"]


class WizardTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT/"app.py"), default_timeout=60).run()

    def assert_clean(self, app):
        self.assertFalse(list(app.exception), str(list(app.exception)))
        self.assertFalse(list(app.error), str([x.value for x in app.error]))

    def test_complete_default_journey(self):
        at = self.app()
        self.assert_clean(at)
        at.button[0].click().run()
        self.assertEqual(at.sidebar.radio[0].value, STEPS[1])
        self.assert_clean(at)
        at.button[0].click().run()
        self.assertEqual(at.sidebar.radio[0].value, STEPS[2])
        self.assert_clean(at)
        self.assertEqual(len(at.metric), 4)
        at.sidebar.radio[0].set_value(STEPS[3]).run()
        self.assert_clean(at)

    def test_settings_persist_across_pages(self):
        at = self.app()
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        at.slider(key="aversion").set_value(10.).run()
        at.number_input(key="rf_rate").set_value(3.5).run()
        at.checkbox(key="include_rf").uncheck().run()
        at.sidebar.radio[0].set_value(STEPS[2]).run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["settings"]["rf_rate"], 3.5)
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        self.assertEqual(at.slider(key="aversion").value, 10.)
        self.assertEqual(at.number_input(key="rf_rate").value, 3.5)
        self.assertFalse(at.checkbox(key="include_rf").value)

    def test_unreachable_target_is_explained(self):
        at = self.app()
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        at.selectbox(key="objective").set_value("达到目标收益").run()
        at.number_input(key="target").set_value(50.).run()
        at.sidebar.radio[0].set_value(STEPS[2]).run()
        self.assertFalse(list(at.exception))
        self.assertTrue(any("不可达" in e.value for e in at.error))

    def test_minimum_variance_cash_has_no_sharpe(self):
        at = self.app()
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        at.selectbox(key="objective").set_value("尽可能降低波动").run()
        at.sidebar.radio[0].set_value(STEPS[2]).run()
        self.assert_clean(at)
        self.assertEqual(at.metric[1].value, "0.00%")
        self.assertEqual(at.metric[2].value, "不适用")

    def test_imported_market_survives_navigation(self):
        from portfolio.data import demo_market, estimate_from_returns
        history = demo_market()[2]
        imported = estimate_from_returns(history.reset_index(), 12)
        imported[0]["name"] = [f"导入资产{i}" for i in range(6)]
        at = self.app()
        at.session_state["imported_market"] = imported
        at.radio(key="source_choice").set_value("导入历史收益 CSV").run()
        self.assert_clean(at)
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        at.sidebar.radio[0].set_value(STEPS[2]).run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["data_source"], "导入历史收益 CSV")
        self.assertTrue(any("导入资产0" in str(df.value) for df in at.dataframe))

    def test_asset_and_bounds_edits_survive_navigation(self):
        at = self.app()
        editor_key = next(k for k in at.session_state.filtered_state if k.startswith("assets_editor_"))
        at.session_state[editor_key] = {"edited_rows": {0: {"预期年收益 (%)": 12.}}, "added_rows": [], "deleted_rows": []}
        at.run()
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        bounds_key = next(k for k in at.session_state.filtered_state if k.startswith("bounds_editor_"))
        at.session_state[bounds_key] = {"edited_rows": {0: {"最高比例 (%)": 30.}}, "added_rows": [], "deleted_rows": []}
        at.run()
        at.sidebar.radio[0].set_value(STEPS[2]).run()
        self.assert_clean(at)
        at.sidebar.radio[0].set_value(STEPS[0]).run()
        self.assertEqual(at.dataframe[0].value.iloc[0]["预期年收益 (%)"], 12.)
        at.sidebar.radio[0].set_value(STEPS[1]).run()
        self.assertEqual(at.session_state["settings"]["bounds"]["中国股票"][1], 30.)


if __name__ == "__main__":
    unittest.main()
