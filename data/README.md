# 中国上市 ETF 行情说明

`cn_etf_prices.csv` 为实际获取的人民币复权收盘价，`cn_etf_snapshot.json` 记录数据源、获取时间、实际窗口、资产代码、币种和 CSV 的 SHA-256。日期为共同有有效价格的交易日，无缺失值填补，不含当天盘中价格。

本次快照由 Yahoo Finance 公开行情接口获取，为 2023-10-25 至 2026-10-23，共 725 个价格观测、724 个日收益观测。可执行 `python scripts/update_market_snapshot.py` 从公开接口重新获取并记录新快照；不要把模拟收益文件改名冒充行情。

| 代码 | 证券名称 | 分析类别 | 证券行情页面 |
| --- | --- | --- | --- |
| 510300 | 沪深300ETF华泰柏瑞 | 权益 | [Yahoo Finance](https://finance.yahoo.com/quote/510300.SS/history/) |
| 510500 | 中证500ETF南方 | 权益 | [Yahoo Finance](https://finance.yahoo.com/quote/510500.SS/history/) |
| 513500 | 标普500ETF博时 | 权益 | [Yahoo Finance](https://finance.yahoo.com/quote/513500.SS/history/) |
| 511010 | 国债ETF国泰 | 债券 | [Yahoo Finance](https://finance.yahoo.com/quote/511010.SS/history/) |
| 511260 | 十年国债ETF国泰 | 债券 | [Yahoo Finance](https://finance.yahoo.com/quote/511260.SS/history/) |
| 518880 | 黄金ETF华安 | 另类 | [Yahoo Finance](https://finance.yahoo.com/quote/518880.SS/history/) |

价格使用数据源的 adjusted close；不使用未经复权的 close 替代。备用来源为东方财富的前复权日收盘价；一次更新对整个候选池使用同一来源。接口均为公开数据服务，可能限流、暂时不可用或发生结构变更。网页失败时显示原因类别并保留真实快照，不自动混入模拟数据；真实快照过旧也会提示。

数据仅用于课堂分析；本项目不提供交易执行、实时委托价格、投资资格或交易状态认证。持仓金额没有进行手数取整。国债 ETF 价格随利率变化，不等同于零波动现金。513500 涉及海外市场、汇率及折溢价；人民币计价不消除这些风险。

估计采用算术日均值乘 252 与样本日协方差乘 252。历史窗口改变可能明显改变结果；这些估计不是未来收益承诺。固定真实候选池用于演示输入与约束，不构成对上述证券的个性化推荐。

可查证的基金说明：[上交所沪深300ETF说明](https://etf.sse.com.cn/fund/learning/download/c/10055614/files/f6353eb4acc542a1b8b0a3a15e858dc9.pdf)、[国泰十年国债ETF资料](https://st.gtfund.com/pis/www/CN_50010000_511260_FA010080_20220001_511260_20221111_090000_01.pdf)、[上交所披露的博时标普500ETF报告](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-07-21/513500_20230721_9OZ5.pdf)。东方财富复权接口格式可参考 [AKShare 官方文档](https://akshare.akfamily.xyz/data/fund/fund_public.html)。
