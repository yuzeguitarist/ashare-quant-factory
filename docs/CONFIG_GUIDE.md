# CONFIG GUIDE

AQF 配置由两部分组成：

- `config.yaml`：非敏感配置（watchlist、调度、GA 参数等）
- `.env`：敏感配置（Gmail App Password 等）

优先级（高 -> 低）：

1. 环境变量（.env / systemd EnvironmentFile）
2. `config.yaml`
3. 默认值（写死在代码里）

## 必填项

- `.env`
  - `AQF_GMAIL_ADDRESS`
  - `AQF_GMAIL_APP_PASSWORD`
  - `AQF_EMAIL_TO`（推荐设置，逗号分隔）

- `config.yaml`
  - `watchlist`：Baostock 格式股票代码列表（如 `sh.600519`）

## 常见调优

- watchlist 太大导致 GA 跑不完：
  - 提高 `ga.workers`（最多 3，4 核建议留 1 核）
  - 降低 `ga.population_size`
  - 设定 `ga.max_eval_symbols`（只用部分股票评估，速度会快很多）

- 邮件太大：
  - 降低 `risk.top_charts`（例如 4）
  - 缩短图表窗口（见 `report.charts` 模块常量）



## Postgres（可选）

- `config.yaml` 新增：
  - `database.backend`: `sqlite` 或 `postgres`
  - `database.url`: Postgres 连接串（仅 backend=postgres 时使用）
- 也可用环境变量覆盖：
  - `AQF_DB_BACKEND=postgres`
  - `AQF_DB_URL=postgresql+psycopg://user:pass@host:5432/aqf`

## 防过拟合设置（GA）

- `ga.cv_method`: `none` / `walk_forward` / `purged_cv`
- `ga.cv_splits`: 时间切片数量
- `ga.cv_purge_days`: Purged CV 每折前后剔除天数
