"""Observable scoring, prompt isolation and paired-study regressions; no model required."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/bench/jevbench"))
import decision_study as study
from make_decision_cases import make_cases


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, route, body):
        self.calls.append((route, body))
        return next(self.responses)


def response(tokens, probabilities):
    return {"model": "test", "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            "choices": [{"logprobs": {"content": [{"candidate_logprobs": [
                {"token": t, "raw_logprob": math.log(p)} for t, p in zip(tokens, probabilities)]}]}}]}


class DecisionStudyTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "task", "family": "test", "state": "Only the second option is supported.",
                    "question": {"type": "choice", "instructions": "Choose.", "criteria": {"second": "Supported", "first": "Unsupported"}},
                    "labels": ["first", "second"], "expected": "second", "provenance": {"oracle": "SECRET GOLD RATIONALE"}}
        self.task = study.Task.read(self.row, "test")

    def test_submission_reuses_exact_prompt_and_shim_preserves_criteria_order(self):
        body, mapping = study.chat_request(self.task, "submitted", "test")
        self.assertEqual(body, study.NInferNativeAdapter(model="test").build_request(self.task))
        self.assertEqual(mapping, {"A": "first", "B": "second"})
        _, shim = study.chat_request(self.task, "shim", "test")
        self.assertEqual(shim, {"A": "second", "B": "first"})
        self.assertNotIn("SECRET GOLD RATIONALE", json.dumps(body))
        native = study.native_request(self.task, "test")
        self.assertEqual(list(native["questions"]["q"]["criteria"]), self.task.labels)

    def test_order_alignment_and_full_two_pass_cost(self):
        a = study.evaluate(self.task, "submitted", "test", FakeClient([response(["A", "B"], [.2, .7])]))
        b = study.evaluate(self.task, "rotated", "test", FakeClient([response(["A", "B"], [.8, .1])]))
        self.assertTrue(a["correct"] and b["correct"])
        self.assertAlmostEqual(a["outside_mass"], .1)
        combined = study.average_orders(self.task, a, b)
        self.assertAlmostEqual(combined["probs"]["second"], (.7 / .9 + .8 / .9) / 2)
        self.assertEqual(combined["latency_s"], a["latency_s"] + b["latency_s"])
        self.assertEqual(combined["usage"]["prompt_tokens"], 20)

    def test_failed_candidates_cannot_become_successful_average(self):
        a = study.evaluate(self.task, "submitted", "test", FakeClient([response(["A", "B"], [.2, .7])]))
        b = study.evaluate(self.task, "rotated", "test", FakeClient([response(["B", "A"], [.8, .1])]))
        self.assertFalse(b["ok"])
        combined = study.average_orders(self.task, a, b)
        self.assertFalse(combined["ok"])
        self.assertEqual(study.metrics([a, b])["accuracy_failures_wrong"], .5)

    def test_reasoning_is_request_capped_and_never_fabricates_probability(self):
        wire = {"model": "test", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 42},
                "content": [{"type": "thinking", "thinking": "A is a tempting distractor."}, {"type": "text", "text": "B"}]}
        client = FakeClient([wire])
        row = study.evaluate(self.task, "reason_1024", "test", client)
        self.assertTrue(row["correct"])
        self.assertIsNone(row["probs"])
        self.assertEqual(client.calls[0][1]["thinking"]["budget_tokens"], 1024)
        self.assertEqual(study.metrics([row])["probability_n"], 0)
        row["usage"]["cache_read_input_tokens"] = 6
        self.assertEqual(study.metrics([row])["mean_input_tokens"], 16)
        wire["stop_reason"] = "max_tokens"
        self.assertFalse(study.evaluate(self.task, "reason_1024", "test", FakeClient([wire]))["ok"])

    def test_ambiguous_generated_answers_and_wrong_model_are_invalid(self):
        for answer in ("A or B", "Answer: B", "B because it is correct", "", "C"):
            with self.assertRaises(ValueError):
                study.parse_answer(answer, {"A": "first", "B": "second"})
        wire = response(["A", "B"], [.2, .7])
        wire["model"] = "different"
        self.assertFalse(study.evaluate(self.task, "submitted", "test", FakeClient([wire]))["ok"])

    def test_paired_metrics_use_same_eligible_tasks_and_keep_failures(self):
        base = {"cohort": "x", "family": "test", "group": "g", "method": "submitted", "latency_s": 1, "probs": None}
        rows = [{**base, "task_id": "a", "ok": True, "correct": False},
                {**base, "task_id": "b", "ok": True, "correct": True},
                {**base, "task_id": "a", "method": "short", "ok": True, "correct": True},
                {**base, "task_id": "b", "method": "short", "ok": False, "correct": False}]
        paired = study.paired(rows)["short"]
        self.assertEqual((paired["fixed"], paired["broken"], paired["delta_accuracy"]), (1, 1, 0))
        self.assertEqual(paired["scenario_groups"], 1)
        # Failed requests can have unknown usage; they must not be reported as zero-cost tokens.
        self.assertEqual(study.metrics(rows)["input_usage_n"], 0)
        self.assertIsNone(study.metrics(rows)["mean_input_tokens"])

    def test_scenario_selection_never_splits_paraphrases(self):
        tasks = [copy.deepcopy(self.task) for _ in range(4)]
        for i, task in enumerate(tasks):
            task.id, task.group = str(i), str(i // 2)
        selected = study.select_tasks(tasks, 1, 42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].group, selected[1].group)

    def test_oracle_cohort_has_distinct_ids_and_valid_semantic_encodings(self):
        tasks = [study.Task.read(row, "oracle") for row in make_cases()]
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len({t.id for t in tasks}), 24)
        for task in tasks:
            if task.question["type"] != "choice":
                body, mapping = study.chat_request(task, "semantic", "test")
                self.assertEqual(set(mapping.values()), set(task.labels))
                self.assertEqual(list(mapping), body["logprob_candidates"])
                self.assertTrue(study.evaluate(task, "rotated", "test", FakeClient([]))["skipped"])

    def test_rule_cohort_covers_rejection_and_waiver_exception(self):
        rules = [row for row in make_cases() if row["family"] == "rule_precedence"]
        self.assertEqual([r["expected"] for r in rules],
                         ["reject_account", "dispatch", "reject_destination", "dispatch"])


if __name__ == "__main__":
    unittest.main()
