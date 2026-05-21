# 回测命令模板

SKILL.md Step 3 触发时再读这份。本文件只放可直接拷贝执行的 bash 范本，不重复 Step 3 的决策流程（哪种模式 / 何时反思 / A/B/C 选择）。

`<SYMBOL>` / `$DATA_CSV` / `<bar数>` / `<资金>` 等占位由 Step 1 / Step 2 已经决定。

---

## 模板 A：不进化模式（一次性完整回测）

```bash
cat > /tmp/bot_params.json << 'PARAMS_EOF'
{完整参数JSON}
PARAMS_EOF

cd {baseDir}/scripts && python3 fetch_data.py --data "$DATA_CSV" --symbol <SYMBOL> --timeframe 15m 2>/dev/null > /tmp/fingerprint.json
CSV_PATH=$(python3 -c "import json; print(json.load(open('/tmp/fingerprint.json'))['csv_path'])")
cd {baseDir}/scripts && python3 run_backtest.py \
  --data "$CSV_PATH" --params-file /tmp/bot_params.json \
  --capital <资金> --output /tmp/backtest_result.json
```

输出：`/tmp/backtest_result.json`，含完整 backtest_result（无 evolution_log）。

---

## 模板 B：进化模式（默认，分四步）

### B1. 保存初始参数 + 生成指纹

```bash
cat > /tmp/bot_params.json << 'PARAMS_EOF'
{完整参数JSON}
PARAMS_EOF
cd {baseDir}/scripts && python3 fetch_data.py --data "$DATA_CSV" --symbol <SYMBOL> --timeframe 15m 2>/dev/null > /tmp/fingerprint.json
```

### B2. 用同一份参数跑分段 baseline（每段都用初始参数）

```bash
CSV_PATH=$(python3 -c "import json; print(json.load(open('/tmp/fingerprint.json'))['csv_path'])")
cd {baseDir}/scripts && python3 run_evolve_backtest.py \
  --data "$CSV_PATH" --params-file /tmp/bot_params.json \
  --segment-bars <bar数> --capital <资金> --output /tmp/evolve_baseline.json
```

输出：`/tmp/evolve_baseline.json`，含 `evolution_log`（每段用同一组参数）。这就是反思素材。

### B3. 反思 → 生成进化计划

读 `cat {baseDir}/knowledge/evolution_guide.md` 拿反思 7 原则；读 `/tmp/evolve_baseline.json` 中的 evolution_log 逐段分析；按以下结构产出进化计划：

```bash
cat > /tmp/evolution_schedule.json << 'EVO_EOF'
[
  {"round": 1, "params": {初始参数}},
  {"round": 2, "params": {反思后调整}},
  ...
]
EVO_EOF
```

注意：`round` 序号必须连续递增；`params` 给出该段使用的**完整参数对象**，不是 patch。

### B4. 用进化计划重跑

```bash
CSV_PATH=$(python3 -c "import json; print(json.load(open('/tmp/fingerprint.json'))['csv_path'])")
cd {baseDir}/scripts && python3 run_evolve_backtest.py \
  --data "$CSV_PATH" --evolution-file /tmp/evolution_schedule.json \
  --segment-bars <bar数> --capital <资金> --output /tmp/evolve_result_final.json
```

输出：`/tmp/evolve_result_final.json`，含按 evolution_schedule 分段执行的最终结果 + 反思后的 evolution_log。**Step 4 上传 verify 用的就是这个文件作 result，params 仍用初始 `/tmp/bot_params.json`**（平台据此做 stitched 回放、与本地结果对比）。
