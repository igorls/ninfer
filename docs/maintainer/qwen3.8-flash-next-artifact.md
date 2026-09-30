# Qwen3.8-Flash-Next v3 artifact

This reference defines the v3 `.ninfer` artifact for Qwen3.8-Flash-Next (transformers
`Qwen4ExpForConditionalGeneration`, text `qwen4_exp_text`), the input of the `qwen4_exp`
architecture package. Framing, numeric semantics and layouts are defined by
[`artifact-container.md`](artifact-container.md), [`tensor-formats.md`](tensor-formats.md) and
[`storage-layouts.md`](storage-layouts.md).

## 1. Derivation

The v3 artifact is derived, never distributed: every consumer rebuilds it from the published v2
artifact with [`tools/convert/qwen4_exp`](../../tools/convert/qwen4_exp).

| Input | Pin |
|---|---|
| v2 artifact | `igorls/Qwen3.8-Flash-Next-mixed-NInfer@5f0ee7e24279cbadf8d4a90c93c2ff5ea6b00688`, `qwen3_8_flash_next_mixed.ninfer`, 113,298,397,952 bytes, SHA256 `3d383e51963aafd4318dfd04c8dc63ee7df11768de19d9ab58dbba44460d1d02` |
| model config | official `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` `config.json`, stored as `tools/convert/qwen4_exp/source_config.json` |
| weight sources of the v2 artifact | `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@a4e813ed` (without the BF16 PLE table) and `primitive-ai/Qwen3.8-Flash-Next-PLE-quant@da8b3958` (`ples_int4`), recipe `qwen3_8_flash_next_mixed-v1` |

```bash
# Linux, unattended and resumable (downloads, upgrades in place, verifies):
python -m tools.convert.qwen4_exp.derive WORKDIR --verify
# Any platform, from an existing v2 file (needs ~102 GiB beside the input):
python -m tools.convert.qwen4_exp.upgrade V2.ninfer V3.ninfer
python -m tools.convert.qwen4_exp.verify V3.ninfer --file-digests
```

`upgrade` reads the v2 file once, in order. That single pass:

1. checks the closed v2 inventory (Section 3) against the official config;
2. copies every object's payload byte for byte into the v3 file set and hashes every v2 object
   range;
3. bakes the two BF16 MTP expert banks into NVFP4 expert banks (Section 4);
4. hashes the whole input and rejects it unless it equals the published SHA256;
5. with `--release-input` (Linux), punches the consumed input range out of the v2 file, so peak
   disk stays about one artifact.

The v3 `artifact_id` is the first 16 bytes of a SHA256 over the v2 digest and the directory
content, so a derivation is reproducible byte for byte on every machine. `verify` re-reads the
result through the generic reader: every object's encoding, size and alignment, every unchanged
object against its v2 digest, and the MTP banks against the pinned loader-oracle digests.
`derive` restarts an interrupted in-place upgrade by discarding the partly released input, because
a hole-punched file keeps its length and would pass a resumed download as complete.

The generic `tools/upgrade_ninfer_v2_to_v3.py` rejects the Flash-Next identity and points here.

The qualified derivation is `artifact_id 1e7e026e9f634926ae26b80f0fc8591e`: four files, 109,681,105,664
bytes, whose SHA256 digests are pinned in `tools/convert/qwen4_exp/source.py` and checked by
`verify --file-digests`. On a Colab G4 an unattended `derive` takes ~13 minutes (6.5 min download,
6.7 min upgrade) plus ~2 minutes of optional verification.

## 2. Components and resources

| Component | Config | Target | Resources |
|---|---|---|---|
| `text` | `Qwen4ExpForCausalLM` / `qwen4_exp_text`, normalized from the official `text_config` | | `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` |
| `vision` | `qwen4_exp_vision`: depth 27, width 1152, intermediate 4304, 16 heads, patch 16, temporal patch 2, merge 2, 2304 positions, `out_hidden_size` 2560 | `text` | `preprocessor_config.json`, `video_preprocessor_config.json` |
| `mtp` | `{"architectures": ["Qwen4ExpMTP"]}` | `text` | |

The text config keeps the facts the architecture interprets: the Qwen dimensions, `layer_types`,
MRoPE (`rope_theta`, `partial_rotary_factor`, `mrope_section`), GDN, MoE (512 experts, top 10,
intermediate 640), hyper-connections (`hc_count` 4, `hc_lowrank` 320), the QSA indexer
(`indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128, `indexer_compress_ratio` 4,
`indexer_budget` 2048) and PLE (`ple_embed_dim`, `ple_conv_kernel_size`, `ngram_size`,
`heads_per_ngram`, `split_ngram_parts`). `ple_layer_ids` is the source's `[2]`, which counts
decoder layers from 1: the PLE weights and injection belong to 0-based layer 1.

The six resources are the v2 artifact's frontend files, unchanged. The chat template is
Flash-Next's official template (8,952 bytes, SHA256 `c3cf9e34abf4...d7a81041`), not the maintained
`tools/chat_templates/qwen3_8.jinja`.

## 3. Objects

The artifact holds 1,566 objects: six resources and 1,560 tensors. Object ids are the v2 names.

| Format | Layout | Objects |
|---|---|---:|
| `bf16` | `contiguous_le_v1` | 1,235 |
| `nvfp4` | `expert_block_scale_k16_m128x4_v1` | 98 (96 text banks + 2 baked MTP banks) |
| `fp8_e4m3fn_row_fp32` | `row_scale_fp32_v1` | 96 |
| `u4z8_g16_fp16` | `packed_u4_g16_v1` | 128 |
| `int64` | `contiguous_le_v1` | 3 |
| resource | `raw_bytes_v1` | 6 |

The logical payload is 109,680,552,704 bytes (102.15 GiB) in four files at the default 32 GB
limit. Every object ahead of the MTP banks keeps its v2 payload offset. The 342 objects stored
after the banks in v2 (MTP attention and norms, then all of Vision, 1.0 GB) move down by the
3.62 GB the NVFP4 banks save.

Per decoder layer `l` (`text/layers/{l}/`), with `H = 2560` and the four-stream width `4H = 10240`:

| Object | Shape | Format |
|---|---|---|
| `{attention,mlp}/hyper_connection/{block_inject,norm,input_mix/down,input_mix/up}` | `[4,10240]`, `[10240]`, `[320,10240]`, `[10240,320]` | bf16 |
| `mlp/router`, `mlp/shared_expert_gate` | `[512,2560]`, `[1,2560]` | bf16 |
| `mlp/shared_expert/{gate,up,down}` | `[640,2560]`, `[640,2560]`, `[2560,640]` | bf16 |
| `mlp/experts/gate_up`, `mlp/experts/down` | `[512,1280,2560]`, `[512,2560,640]` | nvfp4 bank |
| full attention: `attention/query_gate_key_value`, `attention/output` | `[13312,2560]`, `[2560,6144]` | fp8 row fp32 |
| full attention: `attention/{query,key}_norm`, `attention/indexer/query_key`, `attention/indexer/{query,key}_norm` | `[256]`, `[640,2560]`, `[128]` | bf16 |
| GDN: `gdn/query_key_value_z`, `gdn/output` | `[16384,2560]`, `[2560,6144]` | fp8 row fp32 |
| GDN: `gdn/a_b_projection`, `gdn/{a_log,dt_bias}`, `gdn/norm`, `gdn/convolution` | `[96,2560]`, `[48]`, `[128]`, `[4,10240]` | bf16 |

Each query head of `query_gate_key_value` stores `[query_256, output_gate_256]`, followed by 512 key
and 512 value rows. Expert `gate_up` rows are `[gate_640, up_640]`.

Layer 1 also holds PLE: `ple/{key_projection [10240,2560], value_projection [2560,2560],
query_norm, key_norm, conv_norm [10240], convolution [4,10240]}` in bf16, three int64 tables
(`embedding/layer_multipliers [3]`, `embedding/ngram_head_offsets [16]`,
`embedding/ngram_head_vocab_sizes [16]`) and 128 `embedding/shards/{0..127}` of
`[2500012,160]` u4z8 (32,000,161,792 bytes). The shards stay host-resident (mapped page cache).

Globals are `text/token_embedding` and `text/output_head` (`[248320,2560]` bf16, independent) and
the final mixer `text/hyper_connection/{norm,input_mix/down,input_mix/up}`. MTP holds
`mtp/{embedding,hidden}_projection [2560,2560]`, `mtp/embedding_norm [2560]`,
`mtp/hidden_norm [10240]`, its final mixer, and one full-attention MoE layer under `mtp/layer/`
whose attention projections are bf16. Vision is the 333-object Qwen3.5 tower with a 2560-wide merger.

## 4. MTP expert banks

v3 forbids load-time weight repacking, so the upgrade stores the MTP banks in the form the v2
loader built on the device at every start. For each expert `e` with BF16 source `W_e`:

```text
d_e = binary32(2688 / max|W_e|)             (1 for an all-zero expert)
y   = binary32(W_e * d_e)
per 16 consecutive K values:
  s = E4M3FN_RNE(min(binary32(max|y| / 6), 448))
  q = E2M1_RNE(binary32(y / decode(s)))     (saturating at 6; all-zero group when s = 0)
```

This is the repository's `NVFP4_MAXABS_DIVISOR_RNE_V1` encoder
(`tools/convert/quantization/nvfp4.py`) applied per expert, which is the v2 loader kernel
`quantize_nvfp4_expert_bank.cu` at `87812bc8`. The upgrade pins the SHA256 of both baked payloads;
they equal the v2 loader's device banks dumped by `ninfer_quantize_mtp` on G4. An FP64 oracle
also decodes sampled experts from the stored words and confirms the divisor, the scale rule
within one E4M3FN step, and nearest-value codes. The 1,564 other objects are the v2 bytes.

## 5. Bindings and Uses

Bindings are one-to-one with the stored parents under the object id, except:

- `mtp/layer/` becomes `mtp/layers/0/`;
- Vision uses the `qwen3_5` names: `attention/qkv` and `attention/qkv_bias` split into
  `attention/{query,key,value}[_bias]` Parts, `norm{1,2}/{weight,bias}` become
  `norm{1,2}_{weight,bias}`, and the merger norm becomes `vision/merger/norm_{weight,bias}`.

That gives 1,668 bindings. Each projection has one Use per mathematical input position:

| Parameter (per block `B` = `text/layers/{l}/` or `mtp/layers/0/`) | Input |
|---|---|
| `B{attention,mlp}/hyper_connection/{input_mix/down,block_inject}` | `B{attention,mlp}/hyper_connection/normalized_state` |
| `B{attention,mlp}/hyper_connection/input_mix/up` | `B{attention,mlp}/hyper_connection/low_rank` |
| `Battention/query_gate_key_value`, `Battention/indexer/query_key`, `Bgdn/query_key_value_z`, `Bgdn/a_b_projection` | `Bmixer_input` |
| `Battention/output`, `Bgdn/output` | `Battention/gated_output`, `Bgdn/gated_output` |
| `Bmlp/{router,shared_expert_gate,shared_expert/gate,shared_expert/up,experts/gate_up}` | `Bffn_input` |
| `Bmlp/shared_expert/down`, `Bmlp/experts/down` | `Bmlp/shared_expert/product`, `Bmlp/experts/product` |
| `{text,mtp}/hyper_connection/input_mix/{down,up}` | `{text,mtp}/hyper_connection/{normalized_state,low_rank}` |
| `text/layers/1/ple/{key,value}_projection` | `text/layers/1/ple/embedding` |
| `mtp/embedding_projection`, `mtp/hidden_projection` | `mtp/normalized_embedding`, `mtp/normalized_hidden` |
| `text/output_head` | `text/final_hidden`, `mtp/final_hidden` |
| Vision projections | the `qwen3_5` Vision inputs |

The activation policy follows the stored format: `AllowA4` for NVFP4 expert banks, `AllowA8` for
the FP32-scaled FP8 projections, `A16Only` otherwise. The artifact has no activation-divisor
auxiliaries: v2 carried none, so an A4 expert path scales its activations dynamically.

## 6. Loading (`src/models/qwen4_exp`)

The `qwen4_exp` loader consumes this contract exactly. With Vision and MTP selected it binds all
1,668 Bindings and 959 Uses and fails on a missing or unconsumed one; a narrower selection may
leave only `vision/` or `mtp/` entries (and the `text/output_head` Use at `mtp/final_hidden`)
unbound. Residency follows the role:

| Parameters | Residency |
|---|---|
| every projection, norm, embedding, expert bank and Vision weight | `Device`: the stored bytes, 70.01 GiB for Text, 72.33 GiB with Vision and MTP |
| the 128 PLE shards | `Mapped`: page-cache views of the file set, warmed before readiness; a shard that straddles two part files is owned as a copy |
| the three PLE index tables | `Values` (owning INT64), checked as consecutive head row ranges covered by the shards |

`ops::prepare_nvfp4_expert_bank_weight` admits a complete bank with its `AllowA4` Use and rejects a
stored activation divisor, because the A4 route quantizes activations dynamically. The shard height
(2,500,012 rows) is read from the stored shape, not from the config.
