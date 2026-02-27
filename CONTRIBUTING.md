# Contributing

欢迎贡献！如果你想让 AQF 更强：

- 支持更多频率（分钟线）、更多因子（量价/基本面）、更严格撮合（涨跌停/停牌/撮合规则）
- 引入 Postgres/ClickHouse 做全市场数据
- 把 GA 换成 CMA-ES / Bayesian Optimization
- 加入 Walk-forward / Purged CV / 多目标优化（收益/回撤/换手）

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install pytest ruff
```

## 代码规范
- `ruff` 通过
- 重要逻辑需要单元测试
- PR 描述请写清楚：动机、改动点、影响范围

感谢你的贡献！
