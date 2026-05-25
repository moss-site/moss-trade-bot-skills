"""Regression checks for the natural-language skill workflow contract."""

from pathlib import Path
import re
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[2]


class SkillWorkflowContractTest(unittest.TestCase):
    def test_skill_creator_metadata_contract(self):
        skill_md = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---", skill_md, re.DOTALL)
        self.assertIsNotNone(match, "SKILL.md must start with YAML frontmatter")
        frontmatter = match.group(1)
        self.assertIn("name: moss-trade-bot-factory", frontmatter)
        self.assertNotIn("user-invocable", frontmatter)
        self.assertRegex(frontmatter, r"(?m)^name: [a-z0-9-]+$")
        self.assertIn("description:", frontmatter)
        self.assertIn("币种", frontmatter)
        self.assertIn("回测时间区间", frontmatter)
        self.assertIn("策略风格", frontmatter)
        self.assertIn("缺失", frontmatter)
        self.assertIn("upload verify", frontmatter)

    def test_openai_ui_metadata_matches_autonomous_flow(self):
        openai_yaml = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
        self.assertIn('display_name: "Moss Trade Bot Factory"', openai_yaml)
        self.assertIn("short_description:", openai_yaml)
        self.assertIn("$moss-trade-bot-factory", openai_yaml)
        self.assertIn("ETH Livermore-style", openai_yaml)
        self.assertNotIn("ask me to choose", openai_yaml)

    def test_entry_route_resolves_required_fields_or_asks_missing_only(self):
        skill_md = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertLess(skill_md.index("## 入口路由"), skill_md.index("## 数据覆盖发现"))
        for text in (
            "`symbol`",
            "`backtest_range`",
            "`strategy_style`",
            "如果三个必填字段都能从当前消息解析出来",
            "直接继续创建参数并回测",
            "任一必填字段缺失",
            "只询问缺失/无效字段",
            "禁止读取 `params_schema.json`",
            "禁止写 `/tmp/backtest_request.json`",
            "禁止运行 `fetch_data.py`、`run_backtest.py`、`run_evolve_backtest.py`",
            "不要为缺失的币种或策略风格套默认值",
            "用户只说“创建 ETH bot”时，要询问回测区间和策略风格",
        ):
            self.assertIn(text, skill_md)
        for stale_text in (
            "只回复 A/B/C 区间选择问题",
            "必须主动问一次回测区间",
            "回测区间硬门禁",
            "/tmp/backtest_range_decision.json",
            "confirmed_by_user",
            "除回测区间外，直接从描述中推断交易配置",
            "不代表用户已经选择了回测区间，也不允许直接启动回测",
        ):
            self.assertNotIn(stale_text, skill_md)

    def test_dataset_policy_requires_csv_coverage_for_each_symbol(self):
        skill_md = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        data_policy = (SKILL_ROOT / "knowledge" / "data_policy.md").read_text(encoding="utf-8")
        self.assertIn("dataset_catalog.py --symbol", skill_md)
        self.assertIn("dataset_catalog.py --list", skill_md)
        self.assertIn("必须告诉用户每个可用币种的覆盖区间", skill_md)
        self.assertIn("knowledge/data_policy.md", skill_md)
        self.assertIn("不接受用户提供外部 CSV", data_policy)
        self.assertIn("覆盖展示契约", data_policy)
        self.assertIn("每个可用币种的覆盖区间", data_policy)
        self.assertIn("dataset_catalog.py --list", data_policy)

    def test_backtest_template_requires_complete_request_state(self):
        commands_md = (SKILL_ROOT / "knowledge" / "backtest_commands.md").read_text(encoding="utf-8")
        self.assertIn("硬前置条件", commands_md)
        self.assertIn("/tmp/backtest_request.json", commands_md)
        self.assertIn("symbol / strategy_style / data_csv / range_mode / source_text", commands_md)
        self.assertIn("禁止继续写参数或运行回测", commands_md)
        self.assertIn("FETCH_RANGE_ARGS=(--start", commands_md)
        self.assertNotIn("backtest_range_decision", commands_md)
        self.assertNotIn("confirmed_by_user", commands_md)
        self.assertNotIn("confirmed_from_user_reply", commands_md)


if __name__ == "__main__":
    unittest.main()
