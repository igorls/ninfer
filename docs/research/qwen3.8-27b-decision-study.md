# Qwen3.8-27B NVFP4 decision study

23 September 2026. Implementation and initial pilot for experiments 0–3 in the
[JevBench dossier](jevbench-qwen-intelligence-dossier.md). Qwen3.8-27B NVFP4 is the sole target.

Design follow-up: [learned reasoning router and Engine integration](learned-reasoning-router.md).

## Findings

**Bounded reasoning is the strongest quality lead in this pilot.** On 20 selected public hard
decisions, immediate candidate readout was correct on 14; a 1,024-token thinking cap reached 19,
and a 2,048-token cap reached 20. Both reasoning configurations corrected all four baseline errors
on 12 independent synthetic diagnostics. No generated answer was invalid.

**Execution fidelity needs attention before selecting a production policy.** A subsequent probe
found three identical greedy Chat requests changing their answer after an intervening request
published the prefix. Their reported cache reuse changed from 36 tokens to the full prompt.
The pilot's method ordering was randomized, but cache histories were not isolated. Its scores
describe the observed service routes; they do not establish a pure causal effect of reasoning
with every execution detail held fixed.

The study added no engine or serving changes. It used the installed service and preserved the
existing working-tree changes. The implementation is in
[decision_study.py](../../tools/bench/jevbench/decision_study.py), with
[commands and method definitions](../../tools/bench/jevbench/README.md).

## Target and method

- Resident model ID: `qwen3.8-27b`.
- Artifact: `qwen3_8_27b_nvfp4_dflash2.ninfer`, Qwen3.8-27B NVFP4.
- Runtime: installed `ninfer-serve`, FP8 KV, MTP with five draft tokens, LM-head draft enabled,
  Vision loaded, maximum concurrency eight; the study submitted one request at a time.
- Hardware: RTX PRO 6000 Blackwell Workstation Edition. These latency measurements are not RTX
  5090 results and do not claim exclusive GPU access.
- Sample: two scenario groups per family, selected by a fixed hash/seed before outcomes;
  20 JevBench public hard cases across ten families, plus 12 authored exact-answer cases across
  six families. The public tasks are reproduction/development evidence, not untouched sealed data.
- Reasoning: `/v1/messages`, greedy, per-request Engine caps of 1,024/2,048 tokens and 256 additional
  output tokens for the control suffix/answer. The non-thinking Anthropic condition controls for
  that endpoint. Generated answers are strictly parsed, with no fabricated probability metrics.

The smallest supported per-request thinking budget is 1,024 on this endpoint. The study did not
restart the server to test the dossier's proposed 128/512 process defaults. Budget caps are ceilings,
not fixed token expenditures. On public hard tasks the two reasoning conditions used a mean of
578 and 756 total output tokens respectively; the synthetic cases averaged about 215.

## Paired results

Public hard sample, 20 cases unless a smaller applicable subset is shown:

| Method | Correct | Fixed / broken vs submitted | p50 / p95 seconds |
|---|---:|---:|---:|
| Submitted immediate logits | 14 / 20 | — | 0.115 / 0.320 |
| TypeSafe shim ordering | 13 / 20 | 0 / 1 | 0.117 / 0.316 |
| Native System One | 13 / 20 | 1 / 2 | 0.086 / 0.271 |
| Evidence/Criterion/Options framing | 13 / 20 | 1 / 2 | 0.069 / 0.288 |
| Short Chat answer, no thinking | 14 / 20 | 0 / 0 | 0.137 / 0.319 |
| Short Anthropic answer, no thinking | 13 / 20 | 0 / 1 | 0.195 / 0.401 |
| Reasoning cap 1,024 | 19 / 20 | 5 / 0 | 2.188 / 5.310 |
| Reasoning cap 2,048 | 20 / 20 | 6 / 0 | 2.347 / 9.771 |
| Rotated option mapping, choices only | 10 / 15 | 0 / 0 | 0.146 / 0.310 |
| Average two mappings, choices only | 10 / 15 | 0 / 0 | 0.269 / 0.611 |

Two-order latency is the sum of the two separately measured reads. Comparing 10/15 with 14/20
directly would be incorrect; its paired fixed/broken counts use the same fifteen choice tasks.
Native Yes/No or numeric tokens in the framed prompt scored 3/5 on the binary/ordinal subset.

Synthetic diagnostic results were 8/12 for submitted logits, 8/12 for native System One,
7/12 for framing, and 12/12 for each reasoning cap. They cover arithmetic, inclusive dates,
rule precedence, temporal lookup, ordinal rules, and resource constraints. These short authored
cases are mechanism diagnostics, not a representative estimate of general intelligence.

Scenario-bootstrap 95% intervals for the public-hard paired accuracy changes were +10 to +45
percentage points for cap 1,024 and +10 to +50 points for cap 2,048. These small-sample exploratory
intervals do not correct for method selection, public-data exposure, or cache effects. The only
1,024-cap failure was `hard-opus-a-long_policy-04`: it chose `vp_and_finance_director` instead of
`cfo`; the 2,048-cap run answered correctly. One additional case does not settle the best budget.

## Cache/publication probe

Five cases had different generated Chat and Anthropic non-thinking answers in the pilot. The
follow-up replayed each using this sequence twice: Chat read-only, Chat with publication enabled,
and Anthropic non-thinking. Every request and response was retained.

In three cases, the first and second Chat read-only requests had identical JSON, temperature zero,
and the same prompt-token count, but different output letters:

| Task | Initial answer | Later answer | Cached tokens before / after |
|---|---|---|---:|
| `oracle-20260923-inclusive_dates-002` | B | A | 36 / 170 |
| `hard-sol-a-multi_hop-08` | D | B | 36 / 2640 |
| `oracle-20260923-resource_constraints-002` | A | B | 36 / 171 |

The publishing Chat request matched Anthropic on these cases, and the subsequent read-only
request then matched both. This narrows the concern to execution history associated with prefix
publication/reuse. It does **not** identify whether FP8 KV representation, prefill/chunk arithmetic,
checkpoint state, speculative execution, or another cause is responsible. No reference-runtime,
BF16-KV, or operator-oracle comparison was performed.

`prompt_cache_read_only` prevents writes but still permits reads. `prompt_cache_key` is not a
cache namespace in NInfer. Neither flag can establish an isolated cold baseline. Do not recommend
changing either as a quality fix based on this probe.

## Implementation and verification

The runner reuses the submitted adapter's request construction, records exact requests/responses,
validates returned model IDs and candidate mappings, preserves canonical ordinal order, aligns
choice probabilities by semantic ID, and charges both passes for order averaging. Invalid outputs
count as wrong. Probability metrics report their valid-read denominator; missing token usage is
unknown, not zero. Anthropic cached input is included when comparing total input tokens.

Nine focused tests cover submission/shim ordering, gold-data isolation, candidate-order failures,
probability normalization, two-pass accounting, strict generated-answer parsing, cap propagation,
unknown-usage accounting, paired scenario groups, and oracle rule coverage. Python compilation and
whitespace checks passed. Two Luna reviewers separately checked specification and standards.

Review found a missing rule branch in the generated cohort: an active account with a restricted
destination and no waiver. The generator now covers all three decisions and both sides of the
waiver exception. Two supplemental branch cases were run separately; every applicable method
answered both correctly. They are not silently added to the pilot's headline denominators.

Local raw evidence:

- [Pilot manifest](../../profiles/bench/decision-study-20260923/pilot/manifest.json)
- [Pilot results](../../profiles/bench/decision-study-20260923/pilot/results.jsonl)
- [Full pilot metrics](../../profiles/bench/decision-study-20260923/pilot/summary.json)
- [Cache/protocol replay](../../profiles/bench/decision-study-20260923/protocol-control.jsonl)
- [Supplemental branch results](../../profiles/bench/decision-study-20260923/branch-check/report.md)

The profiles directory is local and ignored by Git; the report retains the useful conclusions.

## Next decision

First reproduce and attribute the cache-sensitive answer changes on Qwen3.8-27B NVFP4, then repeat
the promising reasoning comparison with a controlled cache regime. Reasoning at 1,024/2,048 is
worth advancing; this pilot provides no reason to adopt framing or order averaging as a default.
Production changes, calibration of post-reasoning probabilities, training, and 5090 latency
qualification remain outside this completed initial study.
