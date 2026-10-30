# premier-league-prediction — 英超比赛预测 MVP

三条相互独立、最终融合的预测轨道：

| 轨道 | 输入 | 方法 | 输出 |
|---|---|---|---|
| **football**（纯足球） | 历史比分（不含任何赔率特征） | Dixon-Coles + 时间衰减 → 期望进球 → 比分矩阵 | 1X2、TOP2 比分（HT/FT 足球路径留到 v1） |
| **market**（纯市场） | 多家庄赔率 | 1X2 用 Shin 去水，HT/FT（9 维）用乘法去水，跨庄平均 | 1X2、HT/FT（Sportmonks market 29）、TOP2 比分 |
| **ensemble**（集成） | 上述两条轨道的概率 | 对数意见池 p ∝ p_f^w · p_m^(1-w)，w 由历史 log-loss 拟合 | 1X2、HT/FT、TOP2 比分 |

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
- 分页为游标式：读取 `pagination.has_more` / `next_cursor`，下一页传 `cursor=`，**且不能再带 `per_page`**（否则 400）。
- 比分：`scores[].description` 为 `1ST_HALF`（半场比分）、`2ND_HALF`（常规时间结束时的累计比分）、`CURRENT`（最新/最终）；`score.participant` 为 `home`/`away`。主客队取自 `participants[].meta.location`。
- 已完赛状态：`state_id ∈ {5, 7, 8}`；未开赛：`state_id = 1`。
- 限流按实体按小时计（响应体 `rate_limit.remaining / resets_in_seconds`）；遇到 429/5xx 自动指数退避重试，401/403 直接报错（403 通常表示套餐不含 Odds add-on）。

### 赔率快照（opening / T-6h / T-1h）

标准赛前赔率接口只返回**当前**价格，没有历史，所以“开盘价”和“赛前 X 小时”的价格只能靠我们按时抓取并保存。每次 `epl sync odds` 都会追加一份快照到 `odds_snapshots`，并按距开球时间打标签：

- `opening`：该比赛该市场第一次被抓到；
- `T-6h`（≤6.5 小时）、`T-1h`（≤1.5 小时）、`interim`（其他赛前）、`post_kickoff`（开球后，仅留档）。

建议在运营机器上用 cron 调度，例如：每天一次 `epl sync odds --upcoming-days 7`，另外在比赛日每小时跑一次 `epl sync odds --upcoming-days 1`（这样能覆盖 T-6h 与 T-1h）。`epl predict` 默认使用开球前最新的一份快照。

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

指标：log-loss、Brier、RPS、准确率、TOP2 精确比分命中率。最新结果见 [`reports/backtest_epl.md`](reports/backtest_epl.md)。目前的结论：收盘赔率（log-loss ≈ 0.964）优于纯 Dixon-Coles（≈ 0.988），历史拟合得到的足球权重 w≈0，即集成在 1X2 上基本等于市场轨；足球轨的价值主要在于没有赔率时提供独立估计，以及比分分布 / TOP2。

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
    "football": {"p_1x2": [0.46, 0.29, 0.25], "p_htft": null, "top2_scores": [{"score": "1-1", "p": 0.137}, {"score": "1-0", "p": 0.106}],
                 "expected_goals": {"home": 1.48, "away": 1.04}, "model_version": "...", "generated_at": "..."},
    "market":   {"available": true, "p_1x2": [...], "p_htft": [...9...], "top2_scores": [...], "n_bookmakers_1x2": 12, ...},
    "ensemble": {"p_1x2": [...], "p_htft": [...], "top2_scores": [...], "weights": {"football": 0.0, "market": 1.0}, ...}
  }
}
```

TOP2 比分说明：足球轨直接取比分矩阵概率最高的两格；市场轨把去水后的 1X2 反解为 Dixon-Coles 的 (λ, μ) 后取比分；集成轨把足球比分矩阵按集成 1X2 对“主胜/平/客胜”三个区域重新加权后取比分。

## 6. 自我升级钩子（MVP 仅搭架子）

```bash
epl review weekly    # 赛后复盘：已完赛比赛 vs 存储的预测，按轨道计算指标，写 reports/reviews/weekly_*.json
epl review monthly   # 重跑走步回测，刷新报告和集成权重
```

TODO（v1）：每月重新调 ξ / L2、校准曲线、漂移告警、足球轨 HT/FT 路径。

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
