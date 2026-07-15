# 币种杠杆上限

来源：Hyperliquid `/info` meta，按 base asset 查表。`base_leverage` 和 `max_leverage` 都不得超过下表对应币种的 maxLeverage。Step 1 推断杠杆档位（保守 / 中性 / 激进 / 梭哈）后，**先按本表对应 symbol 的上限封顶，再写入参数**；超限不要静默截断，要在 Step 2 摘要中明确告知"已按上限 Nx 封顶"，让用户感知到 cap 存在。

## 当前快照（55 币，2026-07-15 扩展 SMSN/SKHY）

### HyperCore 主板（25 个，USDC 报价）

| maxLeverage | 币种 |
|---|---|
| 40x | BTC |
| 25x | ETH |
| 20x | SOL · XRP |
| 10x | BNB · APT · AVAX · BCH · DOGE · DOT · LINK · LTC · NEAR · SUI · TRX · UNI · ADA · ARB · **HYPE** · **ZEC** · **WLD** |
| 5x  | ATOM · FIL · HBAR · OP |

### xyz HIP-3 builder（30 个，USDC 报价；后端 normalize 自动加 `xyz:` 前缀路由）

| maxLeverage | 币种 |
|---|---|
| 50x | SP500 |
| 30x | XYZ100 |
| 25x | GOLD · SILVER |
| 20x | NVDA · CL · BRENTOIL · **AAPL** · **MSFT** · **TSLA** · **META** · **GOOGL** · **SPCX** · **DRAM** |
| 10x | INTC · AMD · MU · SNDK · MSTR · CRCL · COIN · ORCL · SKHX · CBRS · **TSM** · **MRVL** · **AVGO** · **ZHIPU** · **SMSN** · **SKHY** |

> 上面两组合计 55 币种，与后端 `internal/domain/symbols.go : assetMaxLeverages` 一对一同步（2026-06-08 加入 AAPL/TSM/SPCX；2026-06-15 加入 MSFT 20x、MRVL 10x、AVGO 10x；2026-06-22 加入 ZHIPU 10x，数据源 Gate.io 永续合约 70d，取自 Hyperliquid xyz dex meta；ZHIPU 在 HL 为 onlyIsolated）。2026-06-16 校正 4 个漂移：TSLA/META/GOOGL 10x→20x、SPCX 5x→10x；2026-06-23 SPCX 10x→20x（HL 已上调，经 xyz meta 核实）。2026-06-23 加入 ZEC 10x、WLD 10x（主板 crypto）、DRAM 20x（xyz HIP-3，Roundhill Memory ETF，idx=65，onlyIsolated），均经 Hyperliquid meta 核实。

## 表外币种

如果用户要求的 symbol 不在上面 50 币内：
- **不要假设默认 cap**。Hyperliquid 不同币的实际上限差异大（5x ~ 50x），猜值容易出事
- 优先告诉用户"该币种未在本 skill 的封顶表内，请在创建 bot 前确认 Hyperliquid 实际 maxLeverage"，并默认按 5x 保守处理
- 如果用户坚持指定杠杆，按用户值传给 backend，backend 会按其 risk_config 封顶（详见 backend `agent_trade_risk_configs`）

## 漂移与维护

- Hyperliquid 会按市场流动性调整 maxLeverage（见过 BTC 50x → 40x 的回调）
- 如果上传/创建 bot 接口返回 `leverage exceeds asset cap` 等错误，**以平台返回为准**，不要重试相同杠杆
- 后端代码侧已新增同源 cap 校验（`internal/domain/symbols.go : assetMaxLeverages`），客户端值与本表不一致时优先信任后端

> **2026-07-15 加入 SMSN 10x（三星，跟踪 KRX:005930）、SKHY 10x（SK 海力士 ADS，Nasdaq:SKHY）。**
> ⚠️ **SKHY ≠ SKHX**：SKHX 跟踪 SK 海力士**普通股** KRX:000660，SKHY 跟踪其 **ADS**，1 ADS = 1/10 普通股，且 ADR 长期存在大幅且时变的溢价（2026-07 上市首周从平价扩到 +29%）。两者是**不同标的**，不可互相代理。
> ⚠️ SKHY 的回测数据集 88% 由正股/10 比例回调合成（原生历史不足），详见后端 `share_for_local_run/data_cache/PROVENANCE_SKHY.md`。
> **CXMT（长鑫存储）暂未加入**：标的尚未上市（科创板 688825，2026-07-27 首日交易），xyz 上为 Pre-IPO 永续（价格由 xyz 自有订单簿生成、无外部喂价），全网无可用历史数据。
