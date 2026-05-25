# 数据集覆盖与区间规则

本文件只在需要解释数据覆盖、支持币种、外部 CSV 限制或回测区间时读取。

## 硬规则

- 回测、进化、上传验证只使用本 skill 的 `data_cache_manifest.json` / GitHub Release Asset 明确列出的 Hyperliquid 固定 CSV；`scripts/data_cache/` 只是逻辑路径，真实 CSV 首次运行后位于本地缓存。
- 不接受用户提供外部 CSV，不用一个币种的 CSV 给另一个币种打指纹。
- 覆盖区间以 `dataset_catalog.py` 的输出为准，不以文件名或历史记忆为准。轻量化发布版本不提交 CSV；`dataset_catalog.py` 会在需要时 hydrate Release Asset 后读取真实 CSV。
- 平台是否支持上传或实盘由 Step 4/5 的平台接口返回决定；Step 1 不预先查询平台。

## 数据发现命令

已知 symbol 时：

```bash
SYMBOL="<BTC/USDC 或 ETH/USDC 等>"
cd {baseDir}/scripts && python3 dataset_catalog.py --symbol "$SYMBOL" --timeframe 15m > /tmp/dataset_catalog.json
```

symbol 缺失、用户问“支持哪些币”或需要展示全量覆盖时：

```bash
cd {baseDir}/scripts && python3 dataset_catalog.py --list --timeframe 15m > /tmp/dataset_catalog_all.json
```

输出中：
- `found=false`：没有内置 CSV。展示 `available_symbols` 并请用户重选。
- `found=true`：`csv_path / start / end / bars / compact` 是唯一数据源。
- `--list`：`datasets[]` 是每个可用币种的唯一覆盖清单。

## 覆盖展示契约

- 如果用户已经给出 symbol，只展示该 symbol 的覆盖即可，但必须来自 `dataset_catalog.py --symbol` 的 `start/end/bars`。
- 如果用户没有给出 symbol，必须展示每个可用币种的覆盖区间，帮助用户选择。可以把完全相同覆盖的币种合并成一组，但组内要列出每个币种。
- 如果用户给出的日期超出覆盖，必须用脚本返回的具体 `start/end` 说明边界，然后让用户改选覆盖范围内的日期、最近 N 天或全部可用数据。
- 不要凭文档或记忆里的日期、文件版本（如 `148d`/`304d`）判断覆盖；展示与判断的**具体时段、币种清单一律以 `dataset_catalog.py` 输出为准**。

## 支持的币种与覆盖

本文件**不维护**币种清单或日期（会过时）。需要展示支持哪些币、或某币的可回测区间时，运行：

- 全部币种：`dataset_catalog.py --list` → 以 `datasets[].symbol / start / end / bars` 为准。
- 单个币种：`dataset_catalog.py --symbol <SYMBOL>` → 以 `available_symbols / start / end / bars` 为准。

短历史币种（如部分 xyz 资产上市晚、实际数据短于文件版本号天数）也由脚本读出的真实 `start/end/bars` 反映，不要按文件名里的天数推断。

## 缺数据时的回复

如果用户选择了不在 `available_symbols` 里的 symbol，直接说明：

```text
这个 symbol 当前没有内置固定 CSV，所以不能本地回测，也不能用外部 CSV 替代。
本 skill 当前可选：<available_symbols>。
请从这些 symbol 里重新选一个。
```
