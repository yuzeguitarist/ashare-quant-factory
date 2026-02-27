# ARCHITECTURE

AQF 由一个常驻后台进程（systemd service）驱动，内部包含“夜间流水线”。

## 关键路径

1. **调度器**（Asia/Shanghai）
   - 每天 20:00 触发一次 `NightlyPipeline.run_cycle()`
   - `max_instances=1` 防止重入，另有文件锁兜底

2. **数据更新轮询**
   - 每 5 分钟探测 Baostock 是否已经提供最近交易日的日线数据
   - 探测方式：对 `probe_symbol` 查询该交易日 `date` 对应的日线行是否存在且成交量/收盘价有效

3. **增量入库**
   - SQLite 表 `daily_bars` 以 `(code, date)` 为唯一键
   - 首次拉取：从 `history_start` 到最新交易日
   - 后续拉取：从数据库中该股票的 `max(date)+1` 到最新交易日

4. **遗传算法夜间进化**
   - 在 stop_time（下个交易日开盘前 30 分钟）之前不断迭代
   - 评估函数：对 watchlist 的历史数据跑简化回测，输出复合 fitness（Sharpe/CAGR/DD/turnover）
   - 并行：默认 3 个 workers（4 核机器的甜点位）

5. **建议生成**
   - 用最佳策略在每只股票上计算最近一天信号
   - 输出 buy/sell/hold + 建议权重 + ATR 止损止盈 + 风险评分

6. **HTML 邮件报告**
   - Jinja2 模板渲染
   - mplfinance 生成 K 线与信号图，MIME inline（cid）嵌入邮件
   - 同时落盘到 `data/reports/YYYYMMDD_*.html` 便于审计

## 为什么 SQLite 足够

- watchlist 场景下日线数据量很小（几十只股票 * 数千交易日）
- 单文件、易备份、无外部依赖
- 如果你要扩展到全市场/分钟级，可替换成 Postgres（repository 层已隔离）
