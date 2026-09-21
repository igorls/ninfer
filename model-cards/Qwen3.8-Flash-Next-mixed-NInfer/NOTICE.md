# Attribution and modifications

## Qwen

Qwen3.8-Flash-Next is developed by Qwen. Copyright (c) 2026 Qwen.
The original model and derived weights are subject to the Qwen Community License
1.0 included as `LICENSE`, copied without modification from:

https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE

All six frontend resources embedded in this artifact are byte-identical to the
official repository at that revision. This is a license/frontend reference; it
does not identify the original BF16 weight revision used by the quantization publisher.

## Primitive AI

The compute weights come from `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8`
at `a4e813ed3cfbbcc61e2929699eccb864a4dfa843`. Primitive AI provides the main
NVFP4 expert and FP8 projection quantizations and distributes the remaining BF16
components. Its model card specifies Qwen Community License 1.0.

The INT4 PLE tables come from `primitive-ai/Qwen3.8-Flash-Next-PLE-quant`
at `da8b39586016d8325ac619be28ad77d6296625ec`, using only `ples_int4`.
That repository's model card advertises Apache-2.0. It contains no separate LICENSE
or NOTICE file at the pinned revision. `LICENSE-APACHE-2.0.txt` supplies the Apache
license text alongside this attribution; those metadata do not relicense the
underlying Qwen-derived weights or supersede Qwen's conditions.

## NInfer conversion and runtime

NInfer upstream: https://github.com/Neroued/ninfer

NInfer workstation fork and this release: https://github.com/igorls/ninfer

Changes represented by this release are native `.ninfer` v2 packaging, projection
concatenation/reordering, expert-bank aggregation and scale-layout conversion,
substitution of Primitive's INT4 PLE table for the mixed source's BF16 PLE table,
and embedding of the tokenizer/template/frontend resources. The release performs
no fine-tuning or expert pruning. The 48 main MoE layers retain 512 experts each.

MTP expert banks remain BF16 in the distributed file. When MTP is enabled, the
pinned NInfer loader converts these banks to its NVFP4 execution layout using a
weights-only per-expert/per-group quantizer. This runtime conversion is NInfer's
implementation, distinct from Primitive's main-expert source quantization. It
does not change the published file. No optional pre-spliced NVFP4 MTP artifact is
included in this release.

The historical converter commit was not recorded in the original conversion
report. Exact source revisions, artifact bytes and SHA256, frontend hashes, and a
compatible engine commit are recorded in `artifact-manifest.json`.
