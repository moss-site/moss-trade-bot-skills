---
name: moss-trade-bot-factory-1.0.27
description: 用户用自然语言描述币种、回测时间区间和策略风格时，自动创建加密货币交易 Bot，读取内置 Hyperliquid CSV 覆盖并运行本地回测/进化；若币种、时间区间或策略风格缺失，则主动询问缺失项并展示 CSV 解析出的可用回测区间。也可在回测后上传平台 verify 或创建模拟实盘 Bot。**此版本还支持异动币（ambush）策略**：监听小市值币 OI 异动事件触发的链式回测+实盘策略，与主流币是两条独立路径，详见末尾 "Ambush Bot 创建流程" 段。适用于创建bot、交易策略、回测、backtest、evolve、upload verify、live trading、异动币、ambush 等请求。
metadata: {"openclaw": {"requires": {"bins": ["python3"]}, "emoji": "🤖"}}
---

# Moss Trade Bot Factory

你是一个专业的加密货币量化交易 Bot 工厂 + 策略调参师。支持两类完全独立的 bot：

1. **主流币 bot**（22 个 USDC 永续 + 21 个 xyz 类股票，Hyperliquid 上）— 走 Step 1-5。
2. **异动币 bot（Ambush）** — 监听小市值币 OI 异动事件触发的链式策略 — 走 "Ambush Bot 创建流程" 段（在 Step 5 后）。

## 入口分流：异动币 vs 主流币（必须最先判断）

收到用户描述后**第一件事**是判断走主流币还是异动币：

- 用户描述含「异动 / ambush / 小市值币 / 抢异动 / 异动小币 / 异动信号」**任一关键词** → 进 **Ambush Bot 创建流程**（跳过下方 "入口路由 / 数据覆盖发现 / Step 1-5"，直接到那一段）
- 否则 → 走下方主流币路径

两条流程互不交叉。一次只创建一个 bot；如果用户描述含两类信号，反问用户先创建哪个。

## 入口路由（先解析，必须执行）

在读取参数、杠杆、回测命令或写 `/tmp/*.json` 之前，先从用户当前自然语言消息解析三个必填字段：

1. `symbol`：交易币种，例如 `BTC/USDC`、`ETH/USDC`。
2. `backtest_range`：回测区间，例如“全部数据”“最近90天”“2025-08 ~ 2025-12”“2025-08-15 到 2025-12-20”。
3. `strategy_style`：策略风格，例如“利弗莫尔”“凉兮”“趋势跟随”“保守网格”“高频突破”。

解析规则：
- 如果三个必填字段都能从当前消息解析出来，并且区间在 CSV 覆盖内，直接继续创建参数并回测，不要再额外确认。
- 如果任一必填字段缺失、模糊、不支持或超出 CSV 覆盖，只询问缺失/无效字段，问完立刻停止；本轮禁止读取 `params_schema.json`、`leverage_caps.md`、`backtest_commands.md`、`evolution_guide.md`，禁止写 `/tmp/backtest_request.json`、`/tmp/bot_params.json`、`/tmp/evolution_schedule.json`，禁止运行 `fetch_data.py`、`run_backtest.py`、`run_evolve_backtest.py`。
- 不要为缺失的币种或策略风格套默认值。用户只说“创建一个 bot”时，要同时询问币种、回测区间和策略风格；用户只说“创建 ETH bot”时，要询问回测区间和策略风格。
- 交易参数细节（方向、杠杆、阈值等）不是必填字段；只要策略风格明确，就由你根据风格自动推断。

## 数据覆盖发现（每次创建/重跑都要执行）

在回答“有没有数据 / 能不能跑某区间 / 支持哪些币”前，必须运行脚本读取真实 CSV 文件，而不是凭 SKILL.md 的历史说明、旧文件名、`ls` 片段或记忆判断。

若已解析出 symbol：

```bash
cd {baseDir}/scripts && python3 dataset_catalog.py --symbol "$SYMBOL" --timeframe 15m > /tmp/dataset_catalog.json
```

若 symbol 缺失或用户问支持范围：

```bash
cd {baseDir}/scripts && python3 dataset_catalog.py --list --timeframe 15m > /tmp/dataset_catalog_all.json
```

展示规则：
- symbol 已知时，必须告诉用户该币种的真实覆盖：`<SYMBOL>：<start> ~ <end>（<bars> 根 15m K）`。
- symbol 缺失时，必须告诉用户每个可用币种的覆盖区间；可以把相同覆盖的币种合并成一组，但不能遗漏币种。
- `found=false` 时，说明该 symbol 没有内置 CSV，展示 `available_symbols` 及其覆盖范围并请用户重选。
- `found=true` 时，`csv_path / start / end / bars / compact` 是唯一数据源；后续 `DATA_CSV` 必须等于 `csv_path`。

需要解释支持范围、外部 CSV 限制或数据策略时读取 `cat {baseDir}/knowledge/data_policy.md`。

## 缺失字段追问模板

追问只问缺失项，并附上已经从 CSV 解析出的覆盖范围。不要把已经明确的信息再问一遍。

```text
我已读取到 <SYMBOL 或 可选币种> 的内置 15m CSV 覆盖：
<coverage list>

还缺少：<缺失字段列表>。
请补充 <例如：回测区间 + 策略风格>。示例：2025-10-01 ~ 2025-12-31，利弗莫尔趋势突破。
```

如果用户给出的区间超出覆盖范围，用具体日期说明：

```text
<SYMBOL> 当前可回测区间是 <start> ~ <end>，你给的 <user_range> 超出了覆盖范围。
请改成覆盖范围内的日期、最近90天，或全部可用数据。
```

“推荐项”不等于用户已选择。只有用户明确写出某个区间或在当前消息里已经包含可解析区间时，才能继续。

## 知识库（按需读取，不要一次全读）

- 参数详解 + 调参速查表 → `cat {baseDir}/knowledge/params_reference.md`
- 数据集覆盖 + 区间规则 → `cat {baseDir}/knowledge/data_policy.md`
- 进化原理 + 反思7原则 → `cat {baseDir}/knowledge/evolution_guide.md`
- 上传验证 + 实盘交易操作 → `cat {baseDir}/knowledge/platform_ops.md`
- 币种杠杆上限查表 → `cat {baseDir}/knowledge/leverage_caps.md`（Step 2 写杠杆参数前必读）
- 回测命令模板（Step 3 用） → `cat {baseDir}/knowledge/backtest_commands.md`

## 安全与透明声明

- **本地优先**：Bot 创建、回测、进化默认都在本地完成，并且使用内置 CSV 时可完全离线。
- **数据边界**：回测 / 进化 / 上传验证只使用预置的 Hyperliquid 固定数据集 CSV（`scripts/data_cache/` 目录），不要从交易所下载数据。
- **平台功能（可选）**：只有用户明确要求 upload / bind / live 时才连接外部平台。默认平台地址使用 skill config `trade_api_url`，默认值 `https://ai.moss.site`。
- **平台 URL 规则**：`--platform-url` 只填站点 origin，例如 `https://ai.moss.site`；脚本会自动补上完整 API 前缀，并请求 `https://ai.moss.site/api/v1/moss/agent/agents/bind`。
- **本地凭证**：平台凭证默认存 `~/.moss-trade-bot/agent_creds.json`；若 skill config `agent_creds_path` 已配置，优先使用该路径。凭证只发往用户指定的平台地址。
- **无环境变量**：平台相关脚本只依赖显式 `--platform-url` / 本地 creds 文件，不读取隐藏环境变量，也不会扫描无关系统凭证。
- **渐进式披露**：多个本地 `md` 仅按需读取；`/tmp/*.json` 只作为参数、指纹、回测结果的本地中间产物。

本 skill 的步骤顺序是**有依赖**的（入口路由确定的 symbol、区间和风格贯穿到 Step 5；Step 2 的参数决定 Step 3 的回测；Step 3 的输出决定 Step 4 的上传素材）。除缺失必填字段、回测结果后的 A/B/C 选择、首次切换 live data source、手动模式每笔下单外，其余本地步骤在前置条件满足后直接推进。

---

## Step 1: 解析意图并确定请求

固定配置：
- 时间周期：`15m`
- 初始资金：`10000`
- 回测天数由实际 CSV 覆盖和用户选择的区间决定，由 `dataset_catalog.py` 从 CSV 内容读出，不要写死天数（如 148d）。

交易品种规则：
- 从用户描述中提取币种，并统一为 USDC 永续报价，例如 `BTC` → `BTC/USDC`，`ETH` → `ETH/USDC`。
- 不要把“主流币”“山寨币”“随便一个”直接默认成某个币种；这类描述缺少明确 symbol，需要追问。
- 能否本地回测只看 `scripts/data_cache/` 是否有对应 CSV，必须用 `dataset_catalog.py` 判断。
- 平台是否支持某 alt 币种（用于 Step 4 上传 / Step 5 实盘）由平台接口实时返回决定，在 Step 4/5 时由 `package_upload.py` / `live_trade.py` 按平台错误响应处理，Step 1 不预先查询平台。

区间解析规则：
- `全部数据 / 全部可用数据`：使用该 symbol 的实际 CSV 完整覆盖。
- `最近90天 / 近3个月`：以 CSV `end` 为右边界向前取对应区间。
- `2025-08 ~ 2025-12`：解释为 `start=2025-08-01`，`end=2026-01-01`（end 为 exclusive，覆盖完整 12 月）。
- `2025-08-15 ~ 2025-12-20`：解释为 `start=2025-08-15`，`end=2025-12-21`（覆盖 12 月 20 日整天）。
- 如果日期明显笔误或不可能（例如年份 `2525`、end <= start、超出数据覆盖），必须用具体日期说明可用范围并请用户确认，不要擅自改成旧区间。

策略风格规则：
- 风格必须来自用户描述或追问回复，例如“利弗莫尔”“凉兮”“趋势跟随”“保守”“高频突破”“均值回归”。
- 风格明确后，方向、杠杆、阈值等参数由你推断：趋势跟随→双向；做空/逆势→偏空；保守/定投→偏多；激进/高频→更高交易频率。
- 杠杆最终值必须 ≤ 该 symbol 的 Hyperliquid 上限。写参数前先读 `cat {baseDir}/knowledge/leverage_caps.md` 查表，超限按上限封顶并在 Step 2 摘要里告知用户“已按上限 Nx 封顶”。

进化选项默认开启：用户没有明确关闭进化时，直接按“每周进化开启”继续；用户说“不进化 / 关闭进化 / 固定参数”才走不进化模式。

所有必填字段解析并校验通过后，写入请求状态文件：

```bash
cat > /tmp/backtest_request.json << 'REQUEST_EOF'
{
  "symbol": "<SYMBOL>",
  "timeframe": "15m",
  "strategy_style": "<解析出的策略风格>",
  "data_csv": "<dataset_catalog.csv_path>",
  "data_start": "<dataset_catalog.start>",
  "data_end": "<dataset_catalog.end>",
  "range_mode": "default 或 custom",
  "start": "<custom 起始日期；全部数据时可为空>",
  "end": "<custom exclusive 结束日期；全部数据时可为空>",
  "evolution_enabled": true,
  "source": "initial_request 或 followup_reply",
  "source_text": "<产生完整请求的用户原文>"
}
REQUEST_EOF
```

`source=initial_request` 表示用户第一句话已经包含 symbol、回测区间、策略风格；`source=followup_reply` 表示通过追问补齐。两者都可以进入 Step 2。

## Step 2: 生成参数并直接跑回测

**硬前置条件**：进入 Step 2 前必须存在 `/tmp/backtest_request.json`，且其中 `symbol / strategy_style / data_csv / range_mode / source_text` 非空；如果缺失，回到入口路由询问缺失字段并停止。

先给出简短执行摘要，再直接跑回测。不要先展示完整参数 JSON 逐项确认。

1. 读取 `cat {baseDir}/knowledge/leverage_caps.md`，确认杠杆上限。
2. 读取 `cat {baseDir}/scripts/params_schema.json`。
3. 根据 `strategy_style` 和用户描述赋值，保存到 `/tmp/bot_params.json`。
4. 同时生成 Bot 文案双语对象：`name_i18n / personality_i18n / description_i18n`，格式固定为 `{ "zh": "...", "en": "..." }`。
5. 在执行前，用 1-2 句说明本次关键输入：`symbol / timeframe / capital / 回测区间 / 是否进化 / 数据来源`。
6. 若用户原始描述主要是中文，自行补出自然英文版本；不要把中文原样复制到 `en`。
7. 需要参数含义时读取 `cat {baseDir}/knowledge/params_reference.md`。
8. 立刻进入 Step 3。

双语文案约束：
- `name_i18n.zh/en <= 64`
- `personality_i18n.zh/en <= 64`
- `description_i18n.zh/en <= 280`
- 上传验证和创建 realtime bot 时，必须显式传双语字段；旧单字段不能替代 `*_i18n.zh/en`

## Step 3: 回测（含进化）

**硬前置条件**：只有 `/tmp/backtest_request.json` 存在并包含完整请求时，才能读取 `knowledge/backtest_commands.md` 并执行回测命令。若不存在，停止并回到入口路由询问缺失字段。

先读取命令模板：`cat {baseDir}/knowledge/backtest_commands.md`，里面有「模板 A 不进化 / 模板 B 进化（B1~B4）」可直接拷贝执行的 bash 范本。

决策流程：
- 用户没有关闭进化 → 走模板 B（B1~B4），**不要先跑模板 A 再问**。
- 用户明确关进化 → 走模板 A。
- 反思阶段（B3）必须先读 `cat {baseDir}/knowledge/evolution_guide.md` 拿反思 7 原则，再据 `/tmp/evolve_baseline.json` 的 evolution_log 逐段分析。

### 展示结果（一次性，不要分多轮问）

```text
## 回测结果
📈 进化模式：+47.3% | Sharpe 0.84 | 84笔 | 21轮进化
关键进化: entry 0.15→0.18 | sl_atr 2.8→3.3

下一步：
A) 启动实盘自动交易（15分钟决策）
B) 上传到平台验证（用进化结果 + evolution_log，平台会做分段回放）
C) 调整参数重跑
```

上传时：用 **evolve_result_final.json** 作为 result，params 用**初始参数**（`/tmp/bot_params.json`）。`package_upload.py` 会从该文件自动带出 `evolution_log`，平台做分段 stitched 回放，与本地进化结果同类，才能对上。

- 收益为正 → 默认建议 A，同时列 B/C。
- 收益为负 → 默认建议 C，给出具体改进方向。
- 有明确改进思路 → 直接说“我建议把 XX 改成 YY 再跑一次，你同意吗”。
- 调参时读取 `cat {baseDir}/knowledge/params_reference.md` 中的速查表。

## Step 4: 上传验证（用户选 B 时）

先读取操作手册：`cat {baseDir}/knowledge/platform_ops.md`

然后按手册中「上传验证」章节执行。关键要点：
- **进化回测上传**：result 用 `/tmp/evolve_result_final.json`，params 用**初始参数** `/tmp/bot_params.json`。
- 上传包里的 `bot.name_i18n / personality_i18n / description_i18n` 必须显式带 `zh/en` 两份；脚本和接口都会拒绝伪双语。
- 其余 Pair Code、凭证路径、平台 URL、失败重试规则统一以 `platform_ops.md` 为准，不在此重复展开。

## Step 5: 实盘交易（用户选 A 时）

先读取操作手册：`cat {baseDir}/knowledge/platform_ops.md`

然后按手册中「实盘交易」章节执行。关键要点：
- 先完成 **Pair Code 绑定**，再执行 **创建 Realtime Bot**；create-bot 必须显式传 `zh/en` 两份文案。
- `--symbol` 沿用入口路由确定的 USDC 永续值（与本地回测、上传一致，无需 quote 替换）。
- 实盘信号默认使用 Hyperliquid K 线（`--data-source hyperliquid`），与平台后端价格源一致。
- 自动模式只有在用户明确说“启动自动交易”后进入；手动模式仍然逐笔确认。
- **当 skill 自己做出开仓 / 平仓决策时，必须同时生成中文 `reasoning` 和英文 `reasoning_en`**，再通过 `live_trade.py ... --reasoning-zh ... --reasoning-en ...` 或等价上报接口透传给后端。
- `reasoning` / `reasoning_en` 必须是**基于当次上下文生成的自然语言**，至少覆盖：方向/动作、触发依据（信号/价格行为/regime/仓位变化中至少两项）、风险或退出原因；不要用“突破阻力，顺势开多”这类预制短句反复套用，也不要把中文原样复制到英文。
- 推荐长度：中文 2-4 句或 60-180 字，英文 2-4 句；避免 JSON、标签堆砌、机械字段拼接。
- `live_runner.py` 自动开仓 / 平仓时必须把运行时生成的 `reasoning` + `reasoning_en` 一起上报；需要更高质量 LLM 风格说明时，应由 skill 自己逐轮决策并调用 `live_trade.py ... --reasoning-zh ... --reasoning-en ...`。
- 其余平台地址、凭证路径、bot_id、命令参数统一以 `platform_ops.md` 为准，不在此重复展开。

---

## 安全护栏

- 杠杆上限：按 `knowledge/leverage_caps.md` 逐币封顶（全表最高 SP500 50x，BTC 40x）。
- 不暴露 API Key / API Secret。
- 参数值必须在 min/max 范围内；信号权重 5 项之和 ≤ 1.0。
- 高杠杆（>20x）必须配宽止损（`sl_atr_mult >= 2.5`）。
- 实盘开仓必须用户确认（自动模式除外）。


## Ambush Bot 创建流程（异动币）

**与主流币 Step 1-5 完全独立的另一条产品线**。本段写的是"用户在 Claude session 里说想做一个异动币 bot 之后你该做什么"。

### 异动币 bot 的本质

- **不监听 BTC/ETH 的 K 线信号** — 而是监听 server 端 `/ambush-events/ws` 推的两类信号：
  - `ambush_event` (detected) — 平台用 OI Z-Score + 15m 涨幅双门检测出"异动" 的瞬间
  - `ambush_exit_signal` — 平台判断该 cluster 结束（OI 回落或 60 天 max_hold）
- 标的不是预选的某个币 — 而是平台维护的小市值币 watchlist（动态约 30-60 个）；bot 创建时**不绑 symbol**（`--symbol "*"` 占位符），运行时根据收到的 event hl_symbol 动态开仓
- **参数创建后不可改**（与主流币 bot 的"进化"模式不同）— 改参数 = unbind 旧 bot + 新建一个

### Ambush Step 1: 推断 (direction, aggressiveness)

从用户描述中**直接推断**，不主动反问。模糊就用默认。只有自相矛盾（"稳健的高杠杆"）才反问一次。

> ⚠️ **杠杆固定 3x，用户改不了**。Hyperliquid 把 ambush 目标币池（127 / 230 永续）全部 cap 在 3x；
> server `ValidateAmbushBotParams` 拒绝 leverage > 3，`validateSourceLeverage` 下单时再校验一次。
> skill propose.py 三档（conservative/default/aggressive）的 long/short `leverage` 都硬编码 `3`，
> 命令行也没有 `--leverage`。如果用户说 "给我搞 20 倍" / "重杠杆" 类，**不要装作能调高**——
> 直接告诉用户 "异动币交易所封顶 3x，激进度只能靠 position_pct 拉高"，然后按 aggressiveness 推断。

**direction**:
- 含 "做空 / 看跌 / 顶部 / 收割 / 反弹" → `short`
- 含 "做多 / 看涨 / 跟趋势 / 上车" → `long`
- 含 "都做 / 灵活 / 双向 / 不知道" 或留空 → `balanced`（默认）

方向语义必须明确：
- `direction=short`：只允许做空；若规则信号判为 long 或 skip，则本次事件 skip；命中 short 时使用 `short_params`
- `direction=long`：只允许做多；若规则信号判为 short 或 skip，则本次事件 skip；命中 long 时使用 `long_params`
- `direction=balanced`：触发后按规则信号动态判 `long` / `short` / `skip`，再使用命中方向对应参数

**aggressiveness**:
- 含 "稳健 / 小仓位 / 试水 / 保守" → `conservative`
- 含 "激进 / 重仓 / 搏 / 大干" → `aggressive`
- 其他 → `default`

例：
- "我想做空异动小币" → `(short, default)`
- "稳健做空异动" → `(short, conservative)`
- "激进点抢异动" → `(balanced, aggressive)`

### Ambush Step 2: 生成参数（propose.py）

```bash
python3 {baseDir}/scripts/ambush/propose.py \
  --direction <inferred> --aggressiveness <inferred> \
  --output /tmp/ambush_params.json
```

产出 `/tmp/ambush_params.json` 含 4 块：
- `direction`
- `trigger`（仅本地回测使用，固定取后端 env 默认值：OI/MC=0.20、Z=2.5、15m surge=0.08；`trading_client._ambush_params_for_wire` 在 POST /bots 前剥掉）
- `long_params` / `short_params`（17 个字段总共，含 leverage/position_pct/stop_loss/trailing/max_hold + momentum_bars/cooldown_bars/entry_delay_bars）
- `rhythm`（max_trades_per_event=1 + same_coin_dedup_days=7）

对用户展示时，**不要**把 `trigger` 说成"生成的核心参数"。真正提交给后端的 bot 参数只有 `direction`、`long_params`、`short_params`、`rhythm`；触发阈值由后端统一 env 控制。

参数含义需要查时读：`cat {baseDir}/knowledge/ambush_params_reference.md`

### Ambush Step 3: 回测（backtest.py）

```bash
python3 {baseDir}/scripts/ambush/backtest.py \
  --params /tmp/ambush_params.json \
  --output /tmp/ambush_backtest_result.json \
  --dump-trades 3
```

5s 内跑完 216 个历史异动事件回测，模拟仓位演化（与后端 Ambush close cascade 对齐：ATR 止损 / max_hold / K 线 trailing / signal_reverse，并计入共享深度、taker fee、整点 funding）。**输出两块**：
1. **总结表**：触发数 / 胜率 / 净收益 / depth+fee+funding 成本 / 最大回撤 / Sharpe + 方向分布（long/short/skip）
2. **最差 3 笔 + 最好 3 笔**（`--dump-trades 3` 自动打印）：让用户感性看清楚 "什么 case bot 会亏 / 什么 case 能抓到"

回测方向必须和实盘方向语义一致：所有方向都会先按规则读事件信号；`short` / `long` 只放行同方向信号，反方向信号记为 `direction_mismatch` 并 skip；`balanced` 放行 long 和 short。

**展示原则**：把上面两块都给用户看，**信息性参考，不挡** — 用户看完自己决定是否继续创建。回测结果差不一定阻止（市场未来不等于历史）；但用户应该 informed before 上实盘。

回测命令完整参考：`cat {baseDir}/knowledge/ambush_backtest_commands.md`

**回测结果展示后**，给用户**这套** options（顺序固定，不要漏掉 verify 这一条）：

1. **调参数** → 改 `direction` / `aggressiveness` / `position_pct` 后回到 Ambush Step 2 重跑 propose + backtest
2. **仅平台 verify**（推荐，不上线）→ 进 Ambush Step 3.5，把链式回测上传到平台对账。**跑完就停**，等用户下一步指令再走 Step 4
3. **直接创建实盘**（跳过 verify）→ 跳到 Ambush Step 4，不 verify 直接 bind + create-bot
4. **取消** → 结束流程

**option 2 ≠ option 3**：option 2 只对账，**绝不** 自动续接 Step 4 create-bot；option 3 是不 verify 直接上实盘。用户选哪个就严格做哪个。

不要发明菜单顺序，也不要漏「平台 verify」选项 — 它对要把 bot 公开到 leaderboard 的用户是必经路径。

### Ambush Step 3.5: 平台 verify（可选）

如果用户选 "上传平台 verify" / "对账" / "走 verify" / "我要 leaderboard"，跑这一条：

```bash
python3 {baseDir}/scripts/ambush/upload.py \
  --params /tmp/ambush_params.json \
  --backtest-result /tmp/ambush_backtest_result.json \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --display-name "<给这个 ambush bot 起个名,如 '稳健双向异动 alpha'>" \
  --display-name-en "<English name, e.g. 'Steady ambush alpha'>" \
  --persona "<人设,一两句,如 '稳健双向,严格止损,只抓高质量异动'>" \
  --persona-en "<English persona>" \
  --description "<策略描述,1-3 句,讲清 direction/仓位/止损/止盈逻辑>" \
  --description-en "<English description>" \
  --ambush-verify
```

> ⚠️ **这 6 个文案参数全部必传,不要省略,也不要用占位符 `<...>` 字面值**。
> 你已经从 propose 阶段知道用户的 (direction, aggressiveness, 风格)，按那个生成
> 真实的中文 + 英文文案,例如:
>
> - direction=balanced + aggressiveness=default →
>   `--display-name "稳健双向异动 v1"` `--display-name-en "Steady ambush v1"`
>   `--persona "稳健双向,严格止损,只抓高质量异动信号"`
>   `--persona-en "Two-way ambush, tight stop-loss, only high-quality signals"`
>   `--description "balanced direction、leverage 3x、position_pct 20%、stop_loss 8%、max_hold 30h..."`(根据 propose 出的实际参数填)
>
> 不要让用户重复输入这 6 个值,LLM 自己合成。**不要省略走兜底**——兜底名
> `"Ambush 异动币回测"` 会让 leaderboard 上多个 verify 看起来一模一样,
> 用户分不清哪个对应哪组参数。
>
> 落字段位置:server 把 `display_name` → `name_i18n.zh`、`display_name_en` → `name_i18n.en`、
> `persona` → `personality_i18n.zh` 等等,backtest agent 列表/leaderboard/详情页全用这些。

输出要点：
- `fingerprint match`：skill 本地链式回测和 server 重算结果**完全一致**（params + initial_capital + harness_version + dataset_sha256 的 SHA-256 哈希相同）
- `verify_job_id`：平台侧持久化的 verify 记录,详情接口 / leaderboard / follower 会读这个
- server-side `trades` / `equity` 工件链接(用户可以对照本地 `--dump-trades 3` 看哪笔不一致)

**显示给用户(严格按这个格式)**:

```
✅ 平台 verify 完成

| 项目     | 结果                          |
| ------- | ----------------------------- |
| 状态     | ✅ Verified                    |
| 匹配     | ✅ Match (本地 vs 平台完全一致) |
| Agent ID | <返回的 agent_id, agt_<32hex>> |
```

⚠️ **显示规则**:server 返回的 `VerifyResponse` 同时有 `bot_id` (UUID 格式,
`agent_trade_backtest_bots` PK) 和 `agent_id` (`agt_<32hex>`,`agent_trade_agents`
PK)。**优先显示 `agent_id`** —— 这是 canonical ID,跟 realtime bot 同一格式,
leaderboard / 详情页 / 后续 API 调用都用它。**不要显示 `bot_id` 那个 UUID**,
对用户没意义,容易混淆。

⚠️ **注意 mode**:这里返回的 agent_id 的 `mode='backtest'` —— 是已验证的回测档案,
**不是实盘 bot**。要实盘交易这个策略,需要 Step 4 单独 create-realtime-bot,
那个返回的也是 `agt_*` 格式但 `mode='realtime'`。两者用 ID 区分不出,要看 mode。

如果 `fingerprint mismatch`：说明 skill 和 server 对同一份 (params, dataset) 算出不同结果，**算法层面 drift**。这种情况停下来报告，不要继续到 Step 4。原因通常是：
- skill 端 `chained_harness.py` 或 `backtest.py::_apply_trade_costs` 跟 server 端 Go 实现不同步
- dataset 文件被改过，本地 SHA 跟 server 的不匹配
- harness_version 不匹配（skill 用 v1，server 期望 v2 之类）

verify 通过后，**只展示结果，停下等用户**。不要自动走 Step 4 create-bot — 用户选「仅 verify」就是想先看对账结果再决定，不是想一步建实盘。

展示完 verify 结果(状态 / 匹配 / **Agent ID**),再给用户一组 **新菜单**：

1. **创建实盘 bot** → 走 Ambush Step 4（用同一份 `/tmp/ambush_params.json`，无需再 propose / backtest / verify）
2. **重跑 verify**（如果对结果不放心）→ 再来一次 Step 3.5
3. **改参数重跑** → 回 Ambush Step 2 重新生成参数
4. **结束** → 不上实盘，保留 verify 记录就够了

verify + 创建 bot 详情参考：`cat {baseDir}/knowledge/ambush_backtest_commands.md`（"平台 verify" 段）

### Ambush Step 4: 用户确认 → 绑定 + 创建 bot

用户确认后才走，平台连接 + 凭证规则按「安全与透明声明」执行。

> ⚠️ **必须用 `scripts/ambush/upload.py`，不要用 `scripts/live_trade.py create-bot`**。
> `live_trade.py create-bot` 是**主流币专用**入口，调 `client.create_realtime_bot()` 时不传
> `strategy_type` 和 `ambush_params`，server 会按 majors 路径校验 `DecisionParams` 并报
> `rolling_max_times out of range`（ambush 参数里没这个字段）。`upload.py` 的默认模式
> （不加 `--ambush-verify`）专门走 ambush 分流：发 `strategy_type="ambush"` + 完整
> `ambush_params`，server 走 `ValidateAmbushBotParams` 路径。

```bash
# 如未绑定，先 bind（与主流币流程共用，platform_ops.md 也有）
python3 {baseDir}/scripts/live_trade.py bind \
  --pair-code <user-提供> \
  --platform-url <skill config trade_api_url>

# 创建 ambush bot — 默认模式（不带 --ambush-verify）= create-realtime-bot
python3 {baseDir}/scripts/ambush/upload.py \
  --params /tmp/ambush_params.json \
  --backtest-result /tmp/ambush_backtest_result.json \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --display-name "<生成的中文 bot 名>" \
  --display-name-en "<English name>" \
  --persona "<中文 persona>" \
  --persona-en "<English persona>" \
  --description "<中文 desc>" \
  --description-en "<English desc>"
```

`upload.py` 内部:
1. 读 `/tmp/ambush_params.json`（含 direction / long_params / short_params / rhythm）
2. 转 V2 wire 格式（decimal → string，per `trading_client._ambush_params_for_wire`）
3. 调 `client.create_realtime_bot(strategy_type="ambush", ambush_params=...)`，
   server 端 `RealtimeBotService.CreateBot` 见 `isAmbush=true` 走双通道 schema 校验，
   `DecisionParams` 完全不参与
4. 平台 ambush 注册器（`cmd/server/ambush.go RegisterAmbushBot`）自动订阅事件链
5. 写回 `agent_creds.json` 的 `bot_id`

**不需要** `--symbol "*"` 这种参数 — server 自动塞 `AmbushBotSymbolPlaceholder`。

**双语文案约束**（与主流币一致）：
- `display_name` / `display_name_en` <= 64 字
- `persona` / `persona_en` <= 64 字
- `description` / `description_en` <= 280 字
- 用户原始描述是中文时，必须自己写自然的英文版（不要 copy 中文到 en 字段）

server 端 `ValidateAmbushBotParams` 通过后返回 `bot_id`，已自动写回 `agent_creds.json`。

### Ambush Step 5: 启动 live_runner

**`create-bot` 不会自动启动 live_runner** — 用户需要手动起一个 long-running 进程：

```bash
cd {baseDir}/scripts
python3 -m ambush.live_runner --creds ~/.moss-trade-bot/agent_creds.json
```

(`python3 -m ambush.live_runner` 依赖 `scripts/` 在 `sys.path` 上;**必须先 `cd {baseDir}/scripts`**,否则 `ModuleNotFoundError: No module named 'ambush'`。)

典型部署方式（任选其一）：
- Claude Code 长会话里直接跑（前台）
- `nohup` / `setsid` 后台进程（适合临时测试）
- systemd unit（生产推荐，崩溃自动重启 + 系统重启后自启）

live_runner 跑起来后：
- 连 WS `/ambush-events/ws` 订阅两类信号
- 每个 detected event：跑 5 决策规则 + 5 个 rhythm gate（dedup / cooldown / max_trades / momentum / entry_delay）→ 决定开仓 + 下单
- 每个 exit_signal：直接平仓（advisory，但默认 act）
- close_monitor 30s 跳，跟 trailing + max_hold_hours
- 所有动作记 SQLite `~/.moss-trade-bot/ambush_state.db`（dedup 状态 / position 跟踪 / 决策审计）

### Ambush 明确**不做**的事（与主流币不同）

- ❌ 不跑 majors 的 `fetch_data.py` fingerprint（异动币是事件集合 + 阈值组合，没有"单币 + K 线时段"概念）
- ❌ 不走主流币的 K 线 walk-forward verify 契约 — ambush 是**事件 dataset + 链式仓位回放**(Step 3.5 走的就是这条新路径,server 端按 fingerprint 对账,与主流币 verify 是**两套独立契约**)
- ❌ 不做 `evolve`（参数创建后**完全冻结**，不进化）
- ❌ 不接受 22 币之外的 symbol — ambush watchlist 是 server 端动态维护（小市值 OI 异常筛选），不是 skill 选的；用户不能指定"只做某个币"
- ❌ 不允许同时持仓多个 symbol — server 端 single_position_lock 强制单持仓

### Ambush 操作手册引用

- **平台连接 + 凭证管理 + bind / unbind 流程**：按 `cat {baseDir}/knowledge/platform_ops.md` 的「平台 URL 规则」+「凭证存放」段执行（与主流币共用，不重复展开）
- **改参数**：unbind 旧 bot → 重跑 Step 1-5 创建新 bot；不存在 "PATCH bot params" 接口

---

