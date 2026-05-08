# 平台操作手册（上传验证 + 实盘交易）

## 通用前置

- 凭证存储路径：优先使用 skill config `agent_creds_path`；未配置时默认 `~/.moss-trade-bot/agent_creds.json`（**不要用 /tmp**，重启会丢失）
- 平台地址：优先使用 skill config `trade_api_url`；默认值 `https://ai.moss.site`。也可通过 `--platform-url` 显式传入。首次 bind 后会保存在 `agent_creds.json` 的 `base_url` 字段中，供后续命令复用
- 认证方式：HMAC 签名（api_key + api_secret）
- 下方示例统一写成默认路径；若 skill config 已提供 `agent_creds_path`，请整体替换示例中的凭证文件路径
- `--platform-url` 只填站点 origin，例如 `https://ai.moss.site`。脚本会自动拼成 `https://ai.moss.site/api/v1/moss/agent/agents/bind`

## 依赖声明与无害性

- 平台相关脚本只依赖两类外部输入：显式平台地址（`--platform-url` 或 skill config `trade_api_url`），以及本地 `agent_creds.json` 凭证文件
- `agent_creds.json` 只保存 bind 返回的 `api_key/api_secret`、后续 `bot_id`、以及可选 `base_url`，不包含系统账号或其他第三方密钥
- 这是一项本地文件依赖，不是环境变量依赖；只有在用户明确启用 upload / bind / live 时才会读取
- 回测结果、CSV、参数文件都只在本地读取；只有当你明确执行 upload / bind / live 时，脚本才会向用户指定的平台地址发请求
- 绑定后的本地凭证可直接用于提交并轮询 verify 结果

---

## Pair Code 绑定（上传/实盘的必要前置）

1. **注册**：访问 [Moss Trader](https://moss.site/agent) 注册/登录
2. **获取 Pair Code**：登录后平台显示 **pair code**，用户复制
3. **执行绑定**：
   ```bash
   mkdir -p ~/.moss-trade-bot
   cd {baseDir}/scripts && python3 live_trade.py bind \
     --platform-url "https://ai.moss.site" \
     --pair-code "<pair_code>" \
     --name "<Bot名称>" --persona "<风格>" --description "<策略描述>" \
     --save ~/.moss-trade-bot/agent_creds.json
   ```
4. 返回 `binding_id`、`api_key`、`api_secret`（**bind 仅做身份绑定，不创建实盘 Bot**）。**api_secret 只返回一次，不要打印到回复中。** 若用了 `--save`，同一文件还会保存 `base_url`，供后续命令复用。
5. **实盘前必须再创建 Realtime Bot**（见下「创建 Realtime Bot」），拿到 `bot_id` 写入同一 creds 文件后，才能做 account/positions/orders 等操作。

---

## 上传验证（Step 4）

### 数据要求

平台用 **2025-10-06 ~ 2026-03-03** 区间在服务端回测校验。fingerprint 和 result 必须基于该区间。本地自玩可用其他区间，但上传前需用该区间重跑。

- 回测 / 上传验证使用预置的 Hyperliquid 固定数据集 CSV（`scripts/data_cache/` 目录）
- 上传包中的 Bot 文案现在应显式带双语：`name_i18n/personality_i18n/description_i18n = {zh, en}`
- 上传接口现在按请求体严格校验双语字段：缺任意一个 `*_i18n.zh/en`，即使旧单字段有值，也会直接拒绝
- 若中文文案含中文字符，`package_upload.py` 会要求你补充自然英文版本；不要把中文原样复制到 `en`

### 执行上传前必须确认（缺一不可）

1. 用户已 bind，凭证文件存在
2. 用户明确说「上传」「去传」「提交验证」等

### 重要：平台 verifier 行为

**evolution_log / `--evolution-log-file` 为选填**：接口不强制。**不填 = 不进化模式**（平台用 bot.params 单参回放一整段）；**填了 = 进化模式**（平台按 evolution_log 分段 stitched 回放，与本地 run_evolve_backtest 同类）。

- 本地 `run_backtest.py` / `run_evolve_backtest.py` 与平台 verifier 现在都使用 **全仓回测语义**
- 开仓占用的是账户 `free_margin`
- 强平按账户级 `equity <= maintenance_margin_total` 判定，不再是“单仓亏完自身 margin 就爆”

- **evolution_log 非空**：平台做**分段 stitched 回放**（和本地 run_evolve_backtest 同类），逐段用 evolution_log 里的 params_used，对比你提交的 backtest_result。
- **evolution_log 为空**：平台退化成**单参数普通回放**（只用 bot.params 跑一整段），和本地“分段进化”结果**不是同一类回测**，交易数、收益都会对不上。

因此：**若本次是进化回测，上传必须带 evolution_log**，否则平台按单参回放，本地是分段进化，两边比的不是同一种结果。

### 进化回测上传（推荐：与平台同类对比）

用 **run_evolve_backtest 的输出**作为 result，并带上其中的 evolution_log（脚本可从同一文件自动带出）。params 用**初始参数**（跑进化前的那份）。

```bash
cd {baseDir}/scripts && python3 package_upload.py \
  --bot-name-zh "<中文名称>" \
  --bot-name-en "<English Name>" \
  --bot-personality-zh "<中文风格标签>" \
  --bot-personality-en "<English Personality>" \
  --bot-description-zh "<中文策略描述，≤280字>" \
  --bot-description-en "<English Strategy Description, <=280 chars>" \
  --params-file /tmp/bot_params.json \
  --fingerprint-file /tmp/fingerprint.json \
  --result-file /tmp/evolve_result_final.json \
  --output /tmp/upload_package.json \
  --platform-url https://ai.moss.site \
  --creds ~/.moss-trade-bot/agent_creds.json
```

说明：`evolve_result_final.json` 已含 `evolution_log`，package_upload.py 会从 result 里自动带出，无需再传 `--evolution-log-file`。若显式传，可写 `--evolution-log-file /tmp/evolve_result_final.json`（同文件即可）。

补充说明：

- `--bot-name / --bot-personality / --bot-description` 仍保留做兼容投影
- 新脚本会同时写入 `bot.name_i18n / bot.personality_i18n / bot.description_i18n`
- 若只给中文且未给英文翻译，脚本会直接报错，避免把中文镜像写进 `en`
- 上传验证与轮询只需要本地 `agent_creds.json` 里的 HMAC 凭证

### 固定参数上传（仅当未跑进化时）

若只跑了 run_backtest（未跑进化），则 result 用 run_backtest 的输出，无 evolution_log，平台做单参回放。

### 打包后上传（自动提交 + 轮询，最长120秒）

上述命令已含打包；指定了 `--platform-url` 和 `--creds` 时会自动提交并轮询结果。

### 验证结果处理

- **verified** — 通过，平台自动创建 Agent，告知用户 bot_id
- **rejected** — 不要问用户，自己分析 mismatch_details：
  - 精度问题（偏差 <1%）→ 用 verified_result 替换后重提
  - 数据指纹不匹配 → 重新拉数据生成指纹
  - 差异巨大（>10%）→ 告知用户"平台回测引擎结果有差异"
  - 最多自动重试 2 次
- **failed** — 平台内部错误，稍后重试

### 验证规则

- 数据指纹硬校验：K线数误差 ≤2%，首尾收盘价误差 ≤0.1%
- checksum 不匹配仅警告
- 分段结果容差：2%，总结果容差：1%

---

## 实盘交易（Step 5）

实盘交易完全由载体 LLM 主导，**没有常驻 runner**。每个决策周期：调一次 `advise.py` 拿 advice → 写双语 reasoning → 调 `live_trade.py` 完成下单。三步在同一上下文连贯执行，没有文件中介、没有 advice 过期窗口、没有时钟差。

### advise.py（决策建议 CLI）

```bash
cd {baseDir}/scripts && python3 advise.py \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --params-file /tmp/bot_params.json \
  --symbol BTC/USDT \
  --interval 15
```
默认输出 advice JSON 到 stdout。`--out <path>` 可改写文件（原子写：先 `<path>.tmp` 再 `os.replace`）。advise.py 是 stateless 一次性 CLI——每次调用都拉新行情、查仓位、算信号，不持有任何状态。

#### advice JSON schema

```json
{
  "version": 1,
  "cycle": 1736251200,
  "issued_at":   "2026-05-07T08:30:00Z",
  "valid_until": "2026-05-07T08:45:00Z",
  "symbol": "BTC/USDT", "timeframe": "15m", "data_source": "hyperliquid",
  "action": "open",
  "direction": "LONG",
  "exit_reason": null,
  "context": {
    "mark_price": 65000.0,
    "change_24h_pct": 0.0123,
    "regime": "BULL",
    "signal_value": 1,
    "free_margin": 9876.54,
    "wallet_balance": 10500.0,
    "position": null
  },
  "params_snapshot": {
    "long_bias": 0.5, "base_leverage": 10.0, "max_leverage": 10.0,
    "risk_per_trade": 0.10, "max_position_pct": 0.5,
    "sl_atr_mult": 2.0, "tp_rr_ratio": 3.0,
    "entry_threshold": 0.20, "exit_threshold": 0.10,
    "regime_sensitivity": 0.5
  },
  "suggestion": {
    "leverage": 10,
    "notional_usdt": "1000.00",
    "client_order_id_prefix": "advise-1736251200"
  },
  "reasoning_draft": {
    "zh": "本轮按15m周期评估BTC/USDT，标记价65000.00，24小时上涨1.23%，当前市场状态为BULL，signal_value=1，方向指向开多。账户可用保证金约9876.54 USDT，策略按 risk_per_trade=10.00%、max_position_pct=50.00% 控制敞口，建议使用10x、名义金额1000.00 USDT。执行后以 ATR 止损倍数2.00x和盈亏比3.00R管理风险；若下一周期信号反转、趋势失效或触及止损/止盈，优先减仓或平仓，避免在波动里扩大亏损。",
    "en": "This 15m review for BTC/USDT uses mark 65000.00, a 24h gain of 1.23%, regime BULL, and signal_value=1. These inputs support opening a long position now. Available margin is about 9876.54 USDT, so sizing follows risk_per_trade=10.00% and max_position_pct=50.00%, with suggested 10x leverage and 1000.00 USDT notional. Risk is managed with ATR stop 2.00x and reward/risk 3.00R; if the next cycle shows reversal, regime failure, or stop/take-profit pressure, exposure should be reduced or closed."
  },
  "dispatch_command": [
    "python3", "<abs path>/live_trade.py", "open-long",
    "--creds", "<creds path>",
    "--symbol", "BTC/USDT",
    "--amount", "1000.00",
    "--leverage", "10",
    "--reasoning-zh", "<same as reasoning_draft.zh>",
    "--reasoning-en", "<same as reasoning_draft.en>"
  ]
}
```

`action` 取值：
- `open` → 平仓状态命中信号；`direction` ∈ {LONG, SHORT}；`suggestion` 非空；`dispatch_command` 已带 `open-long`/`open-short` 子命令
- `close` → 持仓命中 `stop_loss` / `take_profit` / `signal_reverse`；`exit_reason` 非空；`dispatch_command` 已带 `close` 子命令
- `hold` → 已持仓无退出信号；`suggestion` / `dispatch_command` 都为 null
- `wait` → 平仓状态无信号；同上

#### 载体 LLM 处理流程
1. `python3 advise.py ...` 拿 advice JSON（stdout 或 --out 文件均可）
2. **按 action 分支**：
   - `open` / `close` → 优先使用 `reasoning_draft.zh/en`；`dispatch_command` 已经带入该草稿，可直接用 subprocess 执行该数组。只有当你能结合额外上下文写得更具体时才改写，但不得缩短成一句话
   - `hold` / `wait` → 不做事

advise.py 是 stateless 的，没有 valid_until 失效需要担心；想要"最新决策"再调一次即可。`cycle` 字段只是 unix epoch 秒，方便你做日志/去重。

#### reasoning 写作约束（必读）
- **必填 + 双语**：`reasoning`（中文）与 `reasoning_en`（英文）都不能空；后端 `_require_bilingual_reasoning` 会校验，缺一拒单
- zh 必须含汉字；en 不能含汉字；各 ≤512 字符；zh 最少 120 字，目标 160–240 字（约 200 字）
- **基于当次上下文真实生成**：至少覆盖方向/动作、触发依据（从 `signal_value / regime / change_24h_pct / position` 中至少引用 2 项）、仓位 sizing、风险或退出原因
- 禁止"突破阻力顺势开多"这类预制短句反复套用；禁止把中文原样翻成英文 placeholder
- 推荐长度：中文 3–5 句 / 160–240 字；英文 3–5 句 / 120–320 字

#### 状态查询
server 是唯一真相。需要时直接查：
```bash
python3 live_trade.py status  --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT
python3 live_trade.py orders  --creds ~/.moss-trade-bot/agent_creds.json
python3 live_trade.py trades  --creds ~/.moss-trade-bot/agent_creds.json
```
**关于 reasoning 字段在哪里能查到**：
- agent-side endpoints `/agent/realtime/bots/{bot_id}/orders` 与 `/fills`（即 `live_trade.py orders` / `trades` 走的接口）**不返回** `reasoning` / `reasoning_en` 字段——这是后端有意省略的（`fillV2Resource` / `orderV2Resource` 不暴露这两列）。
- 想验证 reasoning 是否真的入库 server，去公共 trader detail endpoint：`GET /api/v2/moss/trader/realtime/bots/{bot_id}`，其 `recent_fills[]` 走 `traderRecentFillV2Resource`，包含 `reasoning` 与 `reasoning_en`。
- 跟单只读视角同样从该 trader detail 读，看到的就是这两个字段。

---

### 前置：绑定 + 创建 Realtime Bot

- **绑定**：见上「Pair Code 绑定」，得到 `binding_id`、`api_key`、`api_secret` 并保存到 creds。
- **创建 Realtime Bot**（实盘交易前必须执行一次）：
  ```bash
  cd {baseDir}/scripts && python3 live_trade.py create-bot \
    --creds ~/.moss-trade-bot/agent_creds.json \
    --platform-url "https://ai.moss.site" \
    --symbol "BTC/USDT" \
    --name "<Bot中文名称或默认名称>" \
    --name-zh "<Bot中文名称>" \
    --name-en "<English Bot Name>" \
    --persona "<中文风格标签或默认风格>" \
    --persona-zh "<中文风格标签>" \
    --persona-en "<English Persona>" \
    --description "<中文策略描述>" \
    --description-zh "<中文策略描述>" \
    --description-en "<English Strategy Description>" \
    --params-file /tmp/bot_params.json
  ```
  脚本会把返回的 `bot_id` 写入同一 creds 文件。**多 realtime bot 时**，account/positions/orders 等接口需带 `X-BOT-ID`（本 skill 通过 creds 中的 `bot_id` 自动带上）；若该 binding 下只有一个活跃 bot，服务端可省略。

**unbind 语义**：`unbind` 只**删除当前 realtime bot**（从列表和公开视图移除），**不**吊销 binding 凭证；如需彻底解绑身份，需平台侧另行操作。

### 前置检查

```bash
ls -la ~/.moss-trade-bot/agent_creds.json 2>/dev/null || true
# 如需确认平台地址是否已保存到本地 creds，可读取 base_url：
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".moss-trade-bot" / "agent_creds.json"
if p.exists():
    print(json.load(p.open()).get("base_url", ""))
PY
# creds 中需包含 bot_id（执行过 create-bot 后会有）
```

### 自动运行 Bot

实盘自动运行的入口是 `advise.py + live_trade.py` 串联——见上文「advise.py（决策建议 CLI）」与「载体 LLM 处理流程」两段。每个决策周期由载体 LLM 自己触发，skill 本身**不附带常驻 runner**。下面给出最常用两种接法。

实盘信号默认从 Hyperliquid 永续合约拉取 K 线（`advise.py --data-source hyperliquid`），与平台后端价格源一致。

本周期统一的"任务文案"（替换到下面任一接法里都可以照用）：

> 调用 moss-trade-bot-factory skill 跑一个完整决策周期：先 cd 到 skill scripts 目录，跑 advise.py 拿到 advice JSON；如果 action 是 open/close，优先使用 advice.reasoning_draft.zh/en 和已填好的 advice.dispatch_command，只有能写得更具体时才改写为同等丰满的双语 reasoning，然后 subprocess 执行该数组完成下单。如果 action 是 hold/wait 则本轮不操作。

#### 接法 A：LLM 客户端内置循环（会话开着时最简单）

在你所用的 LLM 客户端会话里调用其循环/计划功能，把上面的"任务文案"作为参数传入。常见入口举例：

| 客户端 | 命令形态 |
|---|---|
| Claude Code | `/loop 15m <任务文案>` |
| ChatGPT | scheduled tasks（每 15 分钟） |
| 其他 agent 框架 | 框架自带的 cron / scheduler |

会话/客户端关掉 → 循环停止。适合白天值守、想随手停的场景。

#### 接法 B：OS 级 cron + LLM CLI（脱离客户端、长期跑）

任何支持"一条命令传入 prompt 并跑一次性 session"的 LLM CLI 都可挂在 cron 上。模板：

```bash
# crontab -e
*/15 * * * * cd /path/to/skill/production/scripts && \
  <YOUR-LLM-CLI> -p "<任务文案>" \
  >> /tmp/moss_cycle.log 2>&1
```

把 `<YOUR-LLM-CLI>` 换成你所用厂商提供的命令行（不同厂商语法不同；一次性执行 + 接受 prompt 参数即可）。每 15 分钟 cron spawn 一次 LLM session 完成本周期，跑完退出。与客户端解耦，机器开着 cron 就跑。

#### 共同建议

- **日志**：把 stdout+stderr 重定向到一个轮转日志文件（`/tmp/moss_cycle.log` 或 `~/.moss-trade-bot/cycle.log`），方便后续排查。
- **去重**：`live_trade.py` 自动用 `client_order_id_prefix=advise-<epoch>` 保证幂等；同一秒不会重复下单。
- **失败兜底**：`advise.py` 网络失败 / `live_trade.py` 校验失败都是非 0 退出码，cron / loop 自动跳过本轮，下个 cycle 重试，无需人工干预。
- **kill switch**：cron 直接 `crontab -e` 注释掉那行；`/loop` 用 `/loop stop` 或终端 Ctrl+C。
- **观测**：`python3 live_trade.py orders --limit 10` 看最近订单；具体 reasoning 入库见前文「状态查询」。

### 手动交易

所有手动交易命令均支持 `--symbol` 参数，默认 `BTCUSDT`。

```bash
cd {baseDir}/scripts

# 查看状态
python3 live_trade.py status --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT

# 做多/做空
python3 live_trade.py open-long --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT --amount 1000 --leverage 10 --reasoning-zh "<由 skill / LLM 按当次信号生成的中文说明>" --reasoning-en "<English decision note generated from the current signal>"
python3 live_trade.py open-short --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT --amount 1000 --leverage 10 --reasoning-zh "<由 skill / LLM 按当次信号生成的中文说明>" --reasoning-en "<English decision note generated from the current signal>"

# 平仓
python3 live_trade.py close --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT --side LONG --reasoning-zh "<由 skill / LLM 按平仓原因生成的中文说明>" --reasoning-en "<English exit note generated from the close reason>"

# 查看历史
python3 live_trade.py orders --creds ~/.moss-trade-bot/agent_creds.json
python3 live_trade.py trades --creds ~/.moss-trade-bot/agent_creds.json
```

### 交易规则

- 支持 `BTCUSDT` 合约，仅市价单
- 杠杆 1-40x
- 下单金额 = `free_margin × risk_per_trade × leverage`
- 开仓前检查 free_margin
- STALE_MARK_PRICE → 等待几秒重试
- 用 `client_order_id` 保证幂等（格式：`{bot_name}-{timestamp}`）
- `open-long` / `open-short` / `close` **必须** 同时传 `--reasoning-zh` 与 `--reasoning-en`；缺一 argparse 直接 exit 2，无 legacy 单语兜底。zh 必须含汉字、en 不能含汉字、各 ≤512；zh 最少 120 字，目标约 200 字。
- 想验证 reasoning 是否真的入库：agent-side `orders` / `trades` 的响应**不会**带 reasoning 字段（`fillV2Resource`/`orderV2Resource` 故意省略），需要去公共 trader detail endpoint `GET /api/v2/moss/trader/realtime/bots/{bot_id}` 看 `recent_fills[]`，那里才暴露 `reasoning` / `reasoning_en`。跟单只读视角看到的也是这一份。

### 安全护栏

**手动模式**：每次开仓前报告方向/金额/杠杆，等用户确认
**自动模式**：用户说"启动自动交易"即为授权，直接启动，不需每笔确认

通用：
- api_secret 不打印到回复
- 启动自动模式前确保用户已看过回测结果并知晓风险
- 发生错误时告知用户
