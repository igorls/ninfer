import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/bench/jevbench"))
import reasoning_router as router


def observation(group="a", **changes):
    row = {"schema": 1, "profile": router.PROFILE, "artifact": "pinned.ninfer", "artifact_sha256": "a" * 64,
           "input": {"id": group, "group": group, "family": "arithmetic",
                     "messages": [{"role": "user", "content": group}],
                     "mapping": {"A": "yes", "B": "no"}, "expected": "yes"},
           "features": [0.25] * 5120,
           "actions": [{"budget": b, "content": "A", "stopped": True,
                        "output_tokens": b + 2} for b in router.BUDGETS]}
    row.update(changes)
    return row


class RouterDataTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return router.load_rows([path], "temporal_lookup")

    def test_truncated_or_extra_text_answers_are_failures(self):
        row = observation()
        row["actions"][0]["stopped"] = False
        row["actions"][1]["content"] = "A because it is correct"
        self.assertEqual(router.labels(row)[0], [0, 0, 1])

    def test_actions_are_independent_not_exclusive(self):
        self.assertEqual(router.labels(observation())[0], [1, 1, 1])

    def test_same_group_stays_together(self):
        a, b = observation("a"), observation("b")
        b["input"]["group"] = a["input"]["group"]
        loaded = self.load([a, b])
        self.assertEqual(loaded[0]["split"], loaded[1]["split"])

    def test_identical_prompts_cannot_cross_splits(self):
        a, b = observation("a"), observation("b")
        b["input"]["family"] = "temporal_lookup"
        b["input"]["messages"] = a["input"]["messages"]
        with self.assertRaisesRegex(ValueError, "cross data splits"):
            self.load([a, b])

    def test_duplicate_or_incomplete_observations_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            self.load([observation(), observation()])
        row = observation()
        row["actions"].pop()
        with self.assertRaisesRegex(ValueError, "complete and ordered"):
            self.load([row])

    def test_nonfinite_features_and_mixed_profiles_fail(self):
        row = observation()
        row["features"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid backbone"):
            self.load([row])
        with self.assertRaisesRegex(ValueError, "incompatible collection"):
            self.load([observation(profile="another-kv-profile")])

    def test_replay_cannot_forget_or_relabel_old_observations(self):
        contract = {"artifact_sha256": "a" * 64}
        parent = {"contract": contract, "training_ids": ["old"],
                  "observation_digests": {"old": "first", "held-out": "second"}}
        router.validate_replay(parent, contract, {"old": "first", "held-out": "second", "new": "third"}, {"old", "new"})
        with self.assertRaisesRegex(ValueError, "previous training example"):
            router.validate_replay(parent, contract, parent["observation_digests"], {"new"})
        with self.assertRaisesRegex(ValueError, "unchanged prior observations"):
            router.validate_replay(parent, contract, {"old": "first", "held-out": "changed"}, {"old"})
        with self.assertRaisesRegex(ValueError, "different feature/training contract"):
            router.validate_replay(parent, {"artifact_sha256": "b" * 64}, parent["observation_digests"], {"old"})

    def test_collection_profile_variants_load_but_never_mix(self):
        for suffix in (":c8", ":derive2048", ":c4:derive2048"):
            self.assertEqual(len(self.load([observation(profile=router.PROFILE + suffix)])), 1)
        with self.assertRaisesRegex(ValueError, "incompatible collection"):
            self.load([observation(profile=router.PROFILE + ":c9")])
        with self.assertRaisesRegex(ValueError, "one collection profile"):
            self.load([observation("a"), observation("b", profile=router.PROFILE + ":c8")])

    def test_analyze_counts_outcomes_identity_and_agreement(self):
        capped = observation("capped")
        capped["actions"][0]["content"] = "B"
        for action, applied in zip(capped["actions"][1:], (True, False)):
            action["thinking"] = {"cap_applied": applied}
        capped["actions"][1]["content"] = "B"
        uncapped = observation("uncapped")
        for action in uncapped["actions"][1:]:
            action.update({"thinking": {"cap_applied": False}, "reasoning": "r", "finish_reason": 3,
                           "output_tokens": 200})
        derived = observation("derived", profile=router.PROFILE + ":derive2048")
        derived["actions"][2]["derived_from_budget"] = 1024
        with tempfile.TemporaryDirectory() as directory:
            data, other, out = (Path(directory) / name for name in ("a.jsonl", "b.jsonl", "r.json"))
            data.write_text("\n".join(json.dumps(r) for r in (capped, uncapped, derived)))
            flipped = json.loads(json.dumps(uncapped))
            flipped["actions"][1]["content"] = "B"
            other.write_text(json.dumps(flipped))
            router.analyze(argparse.Namespace(data=[data], blueprints=None, against=other, out=out))
            report = json.loads(out.read_text())
        entry = report["strata"]["arithmetic"]
        self.assertEqual(entry["n"], 3)
        self.assertEqual(entry["2048_beats_1024"], 1)
        self.assertEqual(entry["reasoning_helps"], 1)
        self.assertEqual(entry["derived_2048"], 1)
        self.assertEqual(entry["cap_applied"], [1, 0])
        self.assertEqual(report["budget_identity_when_1024_uncapped"], {"compared": 1, "identical": 1})
        against = report["against"]
        self.assertEqual((against["shared_ids"], against["features_bitwise_equal"]), (1, 1))
        self.assertEqual(against["label_agreement"], [1, 0, 1])

    def test_same_path_different_weights_cannot_mix(self):
        a, b = observation("a"), observation("b")
        b["artifact_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "one explicitly pinned backbone"):
            self.load([a, b])


if __name__ == "__main__":
    unittest.main()
