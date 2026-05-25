"""Regression test: params_reference.md ranges must match params_schema.json.

Background — 2026-05-11 R1 review (P2-A) found two silently drifted ranges
(exit_threshold doc 0.03~0.30 vs schema 0.05~0.40; trailing_activation_pct
doc 0.01~0.10 vs schema 0.01~0.20). Schema is the runtime source of truth;
the markdown table is what the LLM agent reads. Drift between them mis-trains
the agent (it picks values the schema then rejects, or stays in a window
narrower than what's actually allowed).

This test parses both files and asserts every float-typed param in the table
has its (min, max) verbatim from the schema.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


SKILL_ROOT = os.path.abspath(os.path.join(SCRIPTS, ".."))
SCHEMA_PATH = os.path.join(SCRIPTS, "params_schema.json")
DOC_PATH = os.path.join(SKILL_ROOT, "knowledge", "params_reference.md")


def _parse_doc_ranges():
    """Return {param_name: (low, high)} for every row matching `| name | a~b | ... |`."""
    pattern = re.compile(
        r"^\|\s*([a-z_][a-z0-9_]*)\s*\|\s*([0-9]*\.?[0-9]+)\s*~\s*([0-9]*\.?[0-9]+)\s*\|"
    )
    out = {}
    with open(DOC_PATH, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            name, lo, hi = m.group(1), float(m.group(2)), float(m.group(3))
            out[name] = (lo, hi)
    return out


def _load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # params_schema.json shape: { "version": ..., "categories": ..., "params": { name: {min, max, ...}, ... } }
    return raw.get("params", raw.get("fields", raw))


class ParamsRefDocSchemaParityTest(unittest.TestCase):
    """Every doc-table range must equal the schema's (min, max) for that param."""

    @classmethod
    def setUpClass(cls):
        cls.doc_ranges = _parse_doc_ranges()
        cls.schema_fields = _load_schema()

    def test_doc_parses_at_least_one_param(self):
        self.assertGreater(
            len(self.doc_ranges),
            5,
            f"params_reference.md table parser found {len(self.doc_ranges)} rows; "
            "doc layout may have changed",
        )

    def test_every_doc_range_matches_schema(self):
        mismatches = []
        for name, (doc_lo, doc_hi) in self.doc_ranges.items():
            spec = self.schema_fields.get(name)
            if not spec:
                # Doc lists a name that's not in schema — could be a typo. Flag it.
                mismatches.append(f"{name}: doc has {doc_lo}~{doc_hi} but not in schema")
                continue
            schema_lo = spec.get("min")
            schema_hi = spec.get("max")
            if schema_lo is None or schema_hi is None:
                # bool / enum fields don't have min/max, skip
                continue
            if abs(doc_lo - float(schema_lo)) > 1e-9:
                mismatches.append(
                    f"{name}.min: doc={doc_lo} schema={schema_lo}"
                )
            if abs(doc_hi - float(schema_hi)) > 1e-9:
                mismatches.append(
                    f"{name}.max: doc={doc_hi} schema={schema_hi}"
                )
        if mismatches:
            self.fail(
                "params_reference.md drifted from params_schema.json:\n  - "
                + "\n  - ".join(mismatches)
            )

    def test_known_p2a_pinned(self):
        """Pin the two ranges flagged in 2026-05-11 R1 review §P2-A.

        If schema flips these back to the old narrow ranges (or someone
        rewrites the doc), this fails loudly so we don't re-introduce drift.
        """
        spec = self.schema_fields
        self.assertEqual(spec["exit_threshold"]["min"], 0.05)
        self.assertEqual(spec["exit_threshold"]["max"], 0.40)
        self.assertEqual(spec["trailing_activation_pct"]["min"], 0.01)
        self.assertEqual(spec["trailing_activation_pct"]["max"], 0.20)

        self.assertEqual(self.doc_ranges.get("exit_threshold"), (0.05, 0.40))
        self.assertEqual(self.doc_ranges.get("trailing_activation_pct"), (0.01, 0.20))


if __name__ == "__main__":
    unittest.main()
