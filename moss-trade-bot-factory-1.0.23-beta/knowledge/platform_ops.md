# 平台操作手册（上传验证 + 实盘交易）

## 通用前置

- 凭证存储路径：优先使用 skill config `agent_creds_path`；未配置时默认 `~/.moss-trade-bot/agent_creds.json`（**不要用 /tmp**，重启会丢失）
- 平台地址：优先使用 skill config `trade_api_url`；默认值 `https://beta-api.moss.site`。也可通过 `--platform-url` 显式传入。首次 bind 后会保存在 `agent_creds.json` 的 `base_url` 字段中，供后续命令复用
- 认证方式：HMAC 签名（api_key + api_secret）
- 下方示例统一写成默认路径；若 skill config 已提供 `agent_creds_path`，请整体替换示例中的凭证文件路径
- `--platform-url` 只填站点 origin，例如 `https://beta-api.moss.site`。脚本会自动拼成 `https://beta-api.moss.site/api/v1/moss/agent/agents/bind`

## 依赖声明与无害性

- 平台相关脚本只依赖两类外部输入：显式平台地址（`--platform-url` 或 skill config `trade_api_url`），以及本地 `agent_creds.json` 凭证文件
- `agent_creds.json` 只保存 bind 返回的 `api_key/api_secret`、后续 `bot_id`、以及可选 `base_url`，不包含系统账号或其他第三方密钥
- 这是一项本地文件依赖，不是环境变量依赖；只有在用户明确启用 upload / bind / live 时才会读取
- 回测结果、CSV、参数文件都只在本地读取；只有当你明确执行 upload / bind / live 时，脚本才会向用户指定的平台地址发请求
- 绑定后的本地凭证可直接用于提交并轮询 verify 结果

---

## Pair Code 绑定（上传/实盘的必要前置）

1. **注册**：访问 [Moss Trader](https://beta-api.moss.site/agent) 注册/登录
2. **获取 Pair Code**：登录后平台显示 **pair code**，用户复制
3. **执行绑定**：
   ```bash
   mkdir -p ~/.moss-trade-bot
   cd {baseDir}/scripts && python3 live_trade.py bind \
     --platform-url "https://beta-api.moss.site" \
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
- 因此上传命令里始终要显式带：
  `--bot-name-zh --bot-name-en --bot-personality-zh --bot-personality-en --bot-description-zh --bot-description-en`
- `--bot-name / --bot-personality / --bot-description` 仍会写入单字段，但只是兼容投影，不能代替双语字段

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
  --platform-url https://beta-api.moss.site \
  --creds ~/.moss-trade-bot/agent_creds.json
```

说明：`evolve_result_final.json` 已含 `evolution_log`，package_upload.py 会从 result 里自动带出，无需再传 `--evolution-log-file`。若显式传，可写 `--evolution-log-file /tmp/evolve_result_final.json`（同文件即可）。

补充说明：

- `--bot-name / --bot-personality / --bot-description` 仍保留做兼容投影，便于旧展示位兜底
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

### 前置：绑定 + 创建 Realtime Bot

- **绑定**：见上「Pair Code 绑定」，得到 `binding_id`、`api_key`、`api_secret` 并保存到 creds。
- **创建 Realtime Bot**（实盘交易前必须执行一次）：
  ```bash
  cd {baseDir}/scripts && python3 live_trade.py create-bot \
    --creds ~/.moss-trade-bot/agent_creds.json \
    --platform-url "https://beta-api.moss.site" \
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

```bash
cd {baseDir}/scripts && python3 live_runner.py \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --platform-url "https://beta-api.moss.site" \
  --symbol "BTC/USDT" \
  --params-file /tmp/bot_params.json \
  --interval 15 \
  --log /tmp/bot_live.log
```

> 默认交易对 `BTC/USDT`。

实盘信号从 Hyperliquid 永续合约拉取 K 线（`--data-source hyperliquid`），与平台后端价格源一致。

参数：
- `--interval 15` → 每15分钟决策（对应15m K线）
- `--data-source hyperliquid`（默认）→ 从 Hyperliquid 拉取实时 K 线计算信号
- `--max-cycles 96` → 跑96轮后停（24小时），0=不限
- Ctrl+C 优雅停止

### 手动交易

所有手动交易命令均支持 `--symbol` 参数，默认 `BTCUSDT`。

```bash
cd {baseDir}/scripts

# 查看状态
python3 live_trade.py status --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT

# 做多/做空
python3 live_trade.py open-long --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT --amount 1000 --leverage 10 --reasoning "<由 skill / LLM 按当次信号生成>"
python3 live_trade.py open-short --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT --amount 1000 --leverage 10 --reasoning "<由 skill / LLM 按当次信号生成>"

# 平仓
python3 live_trade.py close --creds ~/.moss-trade-bot/agent_creds.json --symbol BTCUSDT --side LONG --reasoning "<由 skill / LLM 按平仓原因生成>"

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
- `open-long` / `open-short` / `close` 均支持可选 `--reasoning`；该字段应基于当次决策上下文生成，而不是固定模板
- `orders` / `trades` 查询结果会返回 `reasoning`（若该笔订单有上报）；跟单只读视角不会暴露该字段
- `live_runner.py` 自动开仓 / 平仓时会随订单上报一段运行时 `reasoning`；若用户要求更高质量 LLM 风格说明，应由 skill 自己完成逐轮决策并调用 `live_trade.py ... --reasoning ...`

### 安全护栏

**手动模式**：每次开仓前报告方向/金额/杠杆，等用户确认
**自动模式**：用户说"启动自动交易"即为授权，直接启动，不需每笔确认

通用：
- api_secret 不打印到回复
- 启动自动模式前确保用户已看过回测结果并知晓风险
- 发生错误时告知用户
