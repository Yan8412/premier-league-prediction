# premier-league-prediction — 英超比赛预测 MVP

三条相互独立、最终融合的预测轨道：

| 轨道 | 输入 | 方法 | 输出 |
|---|---|---|---|
| **football**（纯足球） | 历史比分（不含任何赔率特征） | Dixon-Coles + 时间衰减 → 期望进球 → 比分矩阵；半场拆分（path A）→ HT/FT | 1X2、HT/FT、TOP2 比分 |
| **market**（纯市场） | 多家庄赔率 | 1X2 用 Shin 去水，HT/FT（9 维）用乘法去水，跨庄平均 | 1X2、HT/FT（Sportmonks market 29）、TOP2 比分 |
| **ensemble**（集成） | 上述两条轨道的概率 | 对数意见池 p ∝ p_f^w · p_m^(1-w)；1X2 与 HT/FT 各自一个权重 | 1X2、HT/FT、TOP2 比分 |

每场比赛、每条轨道输出：`p_1x2[3]`、`p_htft[9]`、`top2_scores[{score,p}×2]`、`model_version`、`generated_at`。

> **使用边界：仅作提示 / 研究参考，不做任何自动下注。**

---

## 1. 安装

需要 Python ≥ 3.10。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # 回测报告使用的精确版本
pip install -e .                  # 安装 `epl` 命令（也可用 python -m epl_prediction）
pytest                            # 全部测试离线运行（HTTP 均为 mock）
```

## 2. Sportmonks 令牌（绝不提交到 git）

令牌**只**从环境变量 `SPORTMONKS_API_TOKEN` 读取：

```bash
export SPORTMONKS_API_TOKEN=你的token
# 或者
cp .env.example .env   # 填入 token；.env 已在 .gitignore 中
```

客户端通过 `Authorization` 请求头发送令牌（Sportmonks 文档支持的两种方式之一），因此令牌不会出现在 URL 或日志中。没有令牌时 `epl smoke` 会打印 SKIP 并以 0 退出，`epl sync ...` 会给出提示并以 2 退出。

```bash
epl smoke   # 有令牌：联赛 8 + 当前赛季 + 未来 14 天第一场比赛的 market 1/29 赔率覆盖情况
```

## 3. 同步 Sportmonks 数据到 SQLite

默认数据库为 `data/epl.db`（已 gitignore，可用 `--db` 或 `EPL_DB_PATH` 修改）。表结构见 `src/epl_prediction/schema.sql`。

```bash
epl sync seasons                         # 联赛 8 的全部赛季 + 当前赛季
epl sync fixtures                        # 当前赛季全部比赛（含比分）
epl sync fixtures --season-id 25583      # 指定赛季
epl sync fixtures --from 2026-10-01 --to 2026-11-30
epl sync fixture --fixture-id 19427xxx   # 单场
epl sync odds --upcoming-days 7          # 未来 7 天比赛的 market 1 + 29 赔率快照
epl sync odds --fixture-id 19427xxx --markets 1,29
epl sync all                             # seasons + 当前赛季 fixtures + 未来 7 天赔率
```

### 使用到的接口、include 与 filter（按 docs.sportmonks.com/v3 核对）

| 用途 | 请求 |
|---|---|
| 赛季 | `GET /v3/football/leagues/8?include=currentSeason;seasons` |
| 赛季详情 | `GET /v3/football/seasons/{id}` |
| 比赛列表 | `GET /v3/football/fixtures/between/{start}/{end}?include=participants;scores;round&filters=fixtureLeagues:8&per_page=50` |
| 单场 | `GET /v3/football/fixtures/{id}?include=participants;scores;round` |
| 赛前赔率 | `GET /v3/football/odds/pre-match/fixtures/{id}/markets/{1\|29}?include=bookmaker` |

- include 用 `;` 分隔；filter 形如 `filters=fixtureLeagues:8`。
- `fixtures/between` 单次最长 100 天，客户端自动切分时间窗。
- 分页：线上 `pagination.next_page` / `next_cursor` 返回的是**完整 URL**。客户端优先从 `next_page` 取 `page` 号（保留 `per_page`）；否则从 `next_cursor` 中提取 `cursor` 令牌（此时**不能**带 `per_page`，否则 400）；绝不把整个 URL 当作 cursor。
- 联赛过滤：除 `filters=fixtureLeagues:8` 外，客户端再严格校验 `league_id == 8`（缺失或其他联赛的比赛会被丢弃并记录 warning）；`--matchday next`、待抓赔率列表、训练历史也都只取 `league_id = 8`。
- 比分：`scores[].description` 为 `1ST_HALF`（半场比分）、`2ND_HALF`（常规时间结束时的累计比分）、`CURRENT`（最新/最终）；`score.participant` 为 `home`/`away`。主客队取自 `participants[].meta.location`。
- 已完赛状态：`state_id ∈ {5, 7, 8}`；未开赛：`state_id = 1`。
- 限流按实体按小时计（响应体 `rate_limit.remaining / resets_in_seconds`）；遇到 429/5xx 自动指数退避重试，401/403 直接报错（403 通常表示套餐不含 Odds add-on）。

### 赔率快照（opening / T-6h / T-1h）

标准赛前赔率接口只返回**当前**价格，没有历史，所以“开盘价”和“赛前 X 小时”的价格只能靠我们按时抓取并保存。每次抓取都会追加一份快照到 `odds_snapshots`，并按距开球时间打标签：

- `opening`：该比赛该市场第一次被抓到；
- `T-6h`：开球前 5–7 小时；`T-1h`：开球前 0.25–1.75 小时；
- `interim`：其他赛前抓取；`post_kickoff`：开球后（仅留档，预测不用）。

**调度器 `epl sync due`**（推荐，每 15 分钟跑一次）：

1. 距上次刷新超过 6 小时则刷新未来 14 天的英超赛程（开球时间可能改动）；
2. 只抓“到期”的比赛：还没有任何快照 → `opening`；进入 T-6h / T-1h 窗口且该标签还没有 → 抓一次；距上次快照超过 24 小时（且开球在 7 小时以后）→ `interim` 刷新；
3. 每轮请求数 = 到期比赛数 × 2 个市场，远低于每实体每小时的限额。

```bash
epl sync due                       # 一次调度 tick（cron 调用它）
epl sync due --horizon-days 10 --refresh-hours 12
epl coverage                       # 查看每场比赛 market 1 / 29 已有哪些快照标签
```

运营机 crontab 示例见 [`deploy/crontab.example`](deploy/crontab.example)（每 15 分钟 `sync due`、每天同步赛程/比分、每天两次 `predict --matchday next`、每天 `audit`）。时间统一用 UTC。`epl predict` 默认使用开球前最新的一份快照（通常就是 T-1h）。

### 数据质量审计 `epl audit`

检查数据库：非 `league_id=8` 的比赛、无法映射到 football-data 队名的球队（需要在 `src/epl_prediction/team_names.csv` 增加别名）、每赛季球队数是否为 20、每轮比赛是否超过 10 场、重复对阵，以及与同赛季 football-data E0 CSV 的球队集合对比。有 error 时退出码为 1。

> 关于 “下一轮出现 Hull / Coventry”：经核对，football-data 2026/27 的 E0（英超）文件中 Coventry、Hull、Ipswich 均为升班马，属于正常英超球队，并非联赛过滤泄漏。为防止真正的泄漏，客户端已改为严格校验 `league_id == 8`（此前 `league_id` 缺失的比赛会被放行），并新增 `epl audit` 供运营机定期核查。

### market 29（HT/FT）覆盖稀疏时的降级

- Sportmonks 文档**没有**列出 market 29 的选项标签格式。解析器接受 `1/X`、`Home/Draw`、`Home - Draw`、`HD` 等写法（同时看 `label` 与 `name`）；无法识别的选项会被跳过，并计入输出 `odds_snapshot["29"].unparsed_selections`，不会去猜。
- 只有 9 个选项全部报价的庄家才参与 HT/FT 共识；没有任何完整庄家时 `p_htft = null` 并在 `notes` 中说明。
- 若 1X2（market 1）缺失但 HT/FT 完整，市场 1X2 由 HT/FT 的全场边际推出（会注明）。
- 若两个市场都没有，市场轨标记 `available: false`，集成轨回退为足球轨。
- `epl sync odds` 会输出每个市场有/无赔率的比赛数，便于监控覆盖率。
- 首次拿到真实令牌后请运行 `epl smoke` 确认 market 29 的真实标签；如有新写法，在 `models/market.py::parse_htft_label` 补充即可。

## 4. 回测（football-data.co.uk，可离线复现）

仓库已提交 2019/20–2026/27 的英超 CSV（`data/football_data/E0_*.csv`，含 FTHG/FTAG/HTHG/HTAG 与开盘/收盘赔率列；2026/27 起还有 HxG/AxG）。

```bash
epl backtest                         # 约 15 秒，写出 reports/backtest_epl.md 与 .json
epl backtest --test-seasons 2425,2526 --name bt_recent
epl data --seasons 2026              # 重新下载/更新某赛季 CSV（起始年份）
```

回测方法（走步，杜绝前视）：

1. 以每个自然周（周一为界）为一轮，用此前所有已赛比赛（最近 1100 天，时间衰减 ξ=0.0019/天）重新拟合 Dixon-Coles；
2. 预测本周比赛；
3. 每家庄的收盘（及开盘）1X2 逐家 Shin 去水后取平均作为市场轨；
4. 第 S 赛季的集成权重 w 只用更早赛季的样本外预测拟合（首个测试赛季用先验 0.5）。

指标：log-loss、Brier、RPS、准确率、TOP2 精确比分命中率；HT/FT 9 类 log-loss / Brier（足球轨 vs 历史频率基线）。最新结果见 [`reports/backtest_epl.md`](reports/backtest_epl.md)。目前的结论：收盘赔率（log-loss ≈ 0.964）优于纯 Dixon-Coles（≈ 0.988），历史拟合得到的足球权重 w≈0，即集成在 1X2 上基本等于市场轨；足球轨的价值主要在于没有赔率时提供独立估计，以及比分分布 / TOP2。

各家庄的列名每个赛季都不同（如 2026/27 没有 Pinnacle），加载器会按列自动识别；`Max*`/`Avg*` 聚合列与 Betfair 交易所列不参与庄家共识。

## 5. 预测

```bash
# 数据库中的比赛（若库里没有且有令牌，会先自动同步该场）
epl predict --fixture-id 19427xxx
epl predict --matchday next -o reports/next_matchday.json   # 下一轮全部未开赛比赛

# 离线模式：不需要令牌 / 数据库
epl predict --home Arsenal --away "Manchester City" --date 2026-11-01 \
            --odds 2.60,3.40,2.75 --odds 2.55,3.50,2.80
```

- 足球轨训练数据 = football-data CSV + 数据库中已完赛的 Sportmonks 比赛（球队名统一为 football-data 写法，映射表见 `src/epl_prediction/team_names.csv`），只使用开球日期之前的比赛。
- 集成权重默认读取 `reports/backtest_epl.json` 中最近赛季的拟合值；`--football-weight 0.3` 可手动覆盖。
- 使用数据库时，预测会写入 `predictions` 表，供每周复盘使用。

输出示例（节选）：

```json
{
  "fixture_id": 19427xxx, "home": "Arsenal", "away": "Man City", "kickoff": "2026-11-01 16:30:00",
  "outcome_order_1x2": ["H", "D", "A"],
  "outcome_order_htft": ["H/H", "H/D", "H/A", "D/H", "D/D", "D/A", "A/H", "A/D", "A/A"],
  "usage": "仅供参考/研究 (tip / research reference only) — no auto-betting.",
  "tracks": {
    "football": {"p_1x2": [0.4585, 0.2864, 0.2551],
                 "p_htft": [0.2913, 0.0594, 0.0193, 0.147, 0.1812, 0.1033, 0.0202, 0.0458, 0.1325],
                 "top2_scores": [{"score": "1-1", "p": 0.137}, {"score": "1-0", "p": 0.106}],
                 "expected_goals": {"home": 1.48, "away": 1.04}, "expected_goals_first_half": {"home": 0.68, "away": 0.42},
                 "consistency": {"htft_ft_margin_max_abs_diff": 0.0}, "model_version": "...", "generated_at": "..."},
    "market":   {"available": true, "p_1x2": [...], "p_htft": [...9...], "top2_scores": [...], "n_bookmakers_1x2": 12, ...},
    "ensemble": {"p_1x2": [...], "p_htft": [...], "top2_scores": [...], "weights": {"football": 0.0, "market": 1.0},
                 "weights_htft": {"football": 0.5, "market": 0.5}, ...}
  }
}
```

### HT/FT（半场/全场）

足球轨的 HT/FT 已上线（原 MVP 的“留到 v1”已取消），采用 path A：

- 与全场共用进攻/防守参数，按“上半场进球占比” s_h / s_a（时间衰减加权拟合，英超近年约主 0.46 / 客 0.40）把期望进球拆成上半场 (λ·s_h, μ·s_a) 与下半场 (λ·(1−s_h), μ·(1−s_a))；
- 上半场比分用轻量 Dixon-Coles（单独拟合的 ρ_ht），下半场为独立泊松，二者卷积得到 (半场比分, 全场比分) 的联合分布，再映射到 9 类（主主 H/H … 客客 A/A）；
- 全场比分边际直接取全场 Dixon-Coles 矩阵，因此足球轨的 `p_htft` 按全场结果求和**严格等于** `p_1x2`，并与 TOP2 比分来自同一比分过程。

集成轨 HT/FT = 足球与市场（market 29）HT/FT 的对数意见池（权重 `weights_htft`，默认先验 0.5，可用 `--football-weight-htft` 覆盖），再按集成 1X2 校正全场边际，保证集成 `p_htft` 与集成 `p_1x2` 一致。没有市场 HT/FT 时集成 HT/FT 取足球轨（会在 notes 注明）。回测中足球 HT/FT 的 9 类 log-loss 约 1.88，优于历史频率基线 1.97（均匀分布 2.20）。

### 输出约定与一致性检查

每条轨道都包含 `p_1x2`、`p_htft`、`top2_scores`、`model_version`、`generated_at`，以及 `consistency.htft_ft_margin_max_abs_diff`（HT/FT 全场边际与 1X2 的最大偏差）。`epl predict` 会用 `epl_prediction.contract.validate_prediction` 自动校验（长度、概率和为 1、TOP2 格式、足球/集成轨 HT/FT 与 1X2 一致），不符合时在 stderr 打印 `contract warning`。市场轨的两个市场来自不同盘口，偏差如实报告、不强行校正。

TOP2 比分说明：足球轨直接取比分矩阵概率最高的两格；市场轨把去水后的 1X2 反解为 Dixon-Coles 的 (λ, μ) 后取比分；集成轨把足球比分矩阵按集成 1X2 对“主胜/平/客胜”三个区域重新加权后取比分。

## 6. 自我升级钩子（MVP 仅搭架子）

```bash
epl review weekly    # 赛后复盘：已完赛比赛 vs 存储的预测，按轨道计算指标，写 reports/reviews/weekly_*.json
epl review monthly   # 重跑走步回测，刷新报告和集成权重
```

TODO（v1）：每月重新调 ξ / L2、校准曲线、漂移告警。

## 7. xG

`src/epl_prediction/data/xg.py` 定义了 `XGProvider` 接口：

- `FootballDataXG`：读取 football-data 自 2026/27 起提供的 `HxG`/`AxG` 列（离线可用）；
- `UnderstatXG`：占位实现（抛 `NotImplementedError`），docstring 中写了抓取方案。Understat 没有官方 API，页面结构不稳定，留到 v1。

MVP 的模型暂不使用 xG。

## 8. 目录结构

```
src/epl_prediction/
  cli.py               # epl 命令
  config.py            # 环境变量 / 路径（令牌只来自环境）
  db.py, schema.sql    # SQLite
  sportmonks/          # client.py（HTTP）、parse.py、sync.py
  data/                # football_data.py、teams.py（队名映射）、xg.py
  models/              # dixon_coles.py、devig.py、market.py、ensemble.py、score_matrix.py
  backtest/            # walkforward.py、metrics.py
  predict.py           # 组装三条轨道的输出
  review.py            # weekly / monthly 钩子
data/football_data/    # 已提交的 football-data CSV（回测可复现）
reports/               # 回测报告（已提交）
tests/                 # pytest；tests/fixtures/ 为按文档结构构造的 Sportmonks 响应样例
```

## 9. 实现参考与许可

全部为自写的独立实现，没有拷贝外部代码。思路参考：Dixon & Coles (1997)；Shin (1992, 1993) 以及 Jullien & Salanié (1994) 的闭式解；开源项目 penaltyblog、mezzala、callumfm/dixon-coles、mberk/shin（MIT）。football-data.co.uk 数据仅用于研究，请遵守其使用条款。
