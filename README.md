# AShare Quant Factory (AQF) — A股夜间策略工厂

> **面向 A 股的“全自动夜间策略进化 + 次日开盘前邮件投递”量化流水线**  
> 数据源：**Baostock 官方接口**｜持久化：SQLite（可平滑迁移到 Postgres）｜策略搜索：遗传算法（GA）｜报告：基金经理风格 HTML 邮件

<p align="center">
  <img alt="AQF" src="https://dummyimage.com/1200x360/111/fff.png&text=AShare+Quant+Factory+(AQF)+%E2%80%94+Nightly+Alpha+Pipeline" />
</p>

---

## 你会得到什么

AQF 在一台普通云服务器（**DigitalOcean 4C/8GB/50GB SSD**）上稳定后台运行：

1. **北京时间 20:00 后自动轮询**（每 5 分钟）检查 Baostock **当日/最近交易日**数据是否已更新  
2. 一旦更新：**自动拉取自选股**（首次全量、后续增量，只补齐数据库里不存在的新数据）并持久化  
3. 自动启动 **遗传算法**，持续进化优化交易策略，直到 **下一个交易日开盘前 30 分钟**停止  
4. 从进化后的种群中智能选出最优策略，生成：
   - 每只股票 **买/卖/持仓**建议
   - **仓位比例**（风控约束下的归一化建议权重）
   - **止损/止盈**（ATR 倍数）
   - **预期收益**、**风险评分**（0–100）
5. 生成**精美 HTML 邮件报告**（含 K 线 + 买卖信号、回测指标、风险仪表盘），通过 **Gmail SMTP** 自动发送到指定邮箱

> ⚠️ 免责声明：本项目仅供研究与教育用途，不构成投资建议。真实交易存在滑点、限价、停牌、涨跌停、T+1 等复杂约束，请自行验证与承担风险。

---

## 一键式开箱即用（Ubuntu 22.04/24.04）

### 0) 服务器建议
- Ubuntu 22.04/24.04
- Python 3.11+
- 4 vCPU / 8GB RAM / 50GB SSD（与你的 DigitalOcean 配置一致）

### 1) 克隆 & 安装
```bash
git clone https://github.com/yourname/ashare-quant-factory.git
cd ashare-quant-factory

bash scripts/bootstrap_ubuntu.sh
```

### 2) 配置自选股与邮件
```bash
cp config.example.yaml config.yaml
cp .env.example .env
nano config.yaml
nano .env
```

- **自选股**：在 `config.yaml` 的 `watchlist` 中填入（Baostock 格式，例如 `sh.600519`）
- **Gmail**：建议开启两步验证并创建 **App Password**，填入 `.env`：
  - `AQF_GMAIL_ADDRESS=你的Gmail`
  - `AQF_GMAIL_APP_PASSWORD=你的AppPassword`
  - `AQF_EMAIL_TO=目标邮箱1,目标邮箱2`

### 3) 初始化数据库（可选，但推荐）
```bash
. .venv/bin/activate
aqf doctor
aqf init-db
```

### 4) 以前台方式试跑一次（推荐）
```bash
aqf run-once
```

### 5) 安装为 systemd 后台服务（生产推荐）
```bash
sudo bash scripts/install_systemd.sh
sudo systemctl status aqf
```

---

## 核心理念（为什么它跑得稳）

- **时区安全**：调度与停止条件严格基于 `Asia/Shanghai`，服务器在美国也不会跑偏
- **增量拉取**：SQLite 以 `(code, date)` 作为唯一键，永不重复写入
- **长任务防重入**：systemd + 文件锁，保证同一时刻只跑一个夜间流水线
- **资源友好**：GA 使用有限工人进程（默认 3 个），保留一个核心给系统与 I/O
- **可观测**：Rich 日志 + 关键事件落库 + 生成的报告 HTML 落盘便于追溯

---

## 功能截图（示意）

> 仓库内不会放你的真实交易数据。你跑起来后，`data/reports/` 会生成每天的 HTML 报告。

- **基金经理风格的日报**：大字号、颜色编码、风险仪表盘
- **嵌入式 K 线 + 信号图**
- **回测指标与参数解释**

---

## CLI 一览

```bash
aqf --help

aqf doctor          # 环境自检（Baostock、DB、邮箱）
aqf init-db         # 初始化 SQLite
aqf run             # 后台调度（建议配合 systemd）
aqf run-once        # 立即跑一次（包含轮询、拉取、进化、出报告、发邮件）
aqf fetch           # 仅拉取并增量入库
aqf evolve          # 仅运行 GA（读取 DB 数据）
aqf report          # 仅生成并发送报告（读取 DB + 最优策略）
```

---

## 项目结构

```text
src/ashare_quant_factory/
  app.py          # 启动调度器（Asia/Shanghai）
  pipeline.py     # 夜间流水线（轮询->拉取->GA->建议->报告->邮件）
  config.py       # YAML + ENV 配置系统
  data/           # Baostock 客户端 & 拉取器
  db/             # SQLite schema & repository
  strategy/       # 指标、回测、遗传算法、建议器
  report/         # 图表、HTML模板、邮件发送
deploy/systemd/   # systemd service 模板
```

---

## 安全与合规

- 不会上传你的数据
- 邮件凭证建议放在 `.env`，并将 `.env` 保持在 `.gitignore`
- 提供 `SECURITY.md` 与最小权限建议

---

## License

Apache-2.0. See `LICENSE`.

---

## Star History

如果你喜欢这个项目，欢迎点个 ⭐️ 并分享给更多量化同好。
