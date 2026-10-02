# Decision policy study tooling

Scripts behind the native-route policy study in the
[decision head plan](../../../../docs/research/decision-head-plan.md): collect the native
`/v1/systemone` readout, its order and letter-rotation variants and a bounded-reasoning answer
over a labeled corpus, then evaluate averaging, calibration and confidence-gated escalation
offline. Python 3.11; the analysis needs `torch` (CPU is enough); the collectors run on a Colab VM
beside a local `ninfer-serve` and keep their `/content` paths.

| Script | Where it runs | What it does |
|---|---|---|
| `build_train_records.py` | VM | Task-form corpus: router synthetic corpus, seeded oracle cases, BoolQ, ANLI, AG News, Yelp; 10% dev split by id hash |
| `collect.py [N]` | VM | Serves the artifact, runs `native`, `native_reversed`, `rotated` on every record and `reason_1024` on the eval sets plus N dev items; one JSON row per call with the full request and response |
| `collect2.py` | VM | Adds the `framed` and `semantic` methods from `decision_study` |
| `analyze_calibration.py COLLECT OUT [BASELINE] [REFERENCE]` | workstation | Raw, global and per-type temperature, and a feature-conditioned argmax-preserving calibrator per averaging mode; fit on train, select on dev, report public; paired rows |
| `analyze_escalation.py COLLECT OUT [BASELINE] [REFERENCE]` | workstation | Escalation gates (confidence, margin, cross-readout disagreement, entropy) at declared dev budgets; reasoning-alone accuracy; paired statistics against the native baseline and the Clef BF16 reference rows |

`BASELINE` is a paired-run `results.jsonl` with `native` rows and `REFERENCE` a `results.jsonl`
with `clef` rows; both default to the recorded runs under `profiles/bench`. Thresholds come from
dev quantiles for a declared budget; the public set is reported, never selected on.
