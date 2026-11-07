# Qwen3.5 架构

Qwen3.5 Dense 与 MoE 共用混合 attention 架构，FFN 使用各自数学实现。
Qwen3.6/3.8 的已注册 27B 和 35B-A3B 是它们的实例；名称与量化组合不创建新的执行入口。
配置和绑定分别由 [config](../../src/models/qwen3_5/config.cpp)与
[load](../../src/models/qwen3_5/load/)维护。

[Engine architecture](engine-architecture.md) defines ownership and execution;
[artifact container](artifact-container.md) defines stored objects and bindings;
[weight conversion](../weight-conversion.md) explains source mapping and recipes.

## Architecture and configuration

The architecture registry recognizes these Text config pairs:

| `architectures[0]` | `model_type` | Feed-forward block |
|---|---|---|
| `Qwen3_5ForCausalLM` | `qwen3_5_text` | Dense SwiGLU |
| `Qwen3_5MoeForCausalLM` | `qwen3_5_moe_text` | Routed SwiGLU experts and a gated shared expert |

Code fixes the pre-norm residual order, gated GQA, zero-centered Text norms, GDN recurrence,
interleaved partial MRoPE, FFN mathematics and component relationships. Config supplies instance
dimensions and layer choices. It does not describe a graph of arbitrary operators.

[`config.cpp`](../../src/models/qwen3_5/config.cpp) defines the exact fields and validation.
Their responsibilities are:

| Config facts | Meaning |
|---|---|
| `hidden_size`, `vocab_size`, `num_hidden_layers`, `max_position_embeddings` | Mathematical extents |
| `layer_types` | One `full_attention` or `linear_attention` entry per Text block |
| `tie_word_embeddings`, `rms_norm_eps` | Training relationship and normalization constant |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | GQA dimensions; also used by MTP |
| `rope_parameters` | Theta, partial rotary factor and three interleaved MRoPE sections |
| `linear_num_*_heads`, `linear_*_head_dim`, `linear_conv_kernel_dim` | GDN heads, state dimensions and convolution width |
| `intermediate_size` | Dense FFN width |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` | MoE dimensions |

Projection widths, compact attention/GDN layer indices, rotary dimensions and parameter shapes are
derived. The converter expands source shorthand such as a full-attention interval into the explicit
`layer_types` array. Source dtype, quantization metadata, checkpoint name and training-only fields
do not become mathematical config.

Text is required. Optional components target `text` and supply their private data:

- Vision supplies tower dimensions and patch/merge geometry. Its output width comes from Text.
- MTP identifies `Qwen3_5MTP` or `Qwen3_5MoeMTP`. The single-layer predictor inherits Text's
  attention, RoPE and FFN geometry and shares its embedding/output semantics.
- DFlash/DFlash2 supply private draft dimensions, attention pattern, conditioning taps and their
  algorithm parameters. See [DFlash](dflash.md).

Only selected components are semantically bound and materialized. A model's weights and config
remain immutable after loading; request capacity, concurrency, KV codecs and graph profiles are
runtime choices.

### Current official geometries

These are useful workload points, rather than a second config registry:

| Quantity | 27B Dense | 35B-A3B MoE |
|---|---:|---:|
| Hidden width H | 5120 | 2048 |
| Text layers | 64 | 40 |
| Full attention / GDN layers | 16 / 48 | 10 / 30 |
| Q heads / KV heads / head dimension | 24 / 4 / 256 | 16 / 2 / 256 |
| GDN key heads / value heads | 16 / 48 | 16 / 32 |
| GDN key/value head dimension | 128 / 128 | 128 / 128 |
| GDN convolution width | 4 | 4 |
| Dense intermediate width | 17408 | — |
| Routed experts / selected per token | — | 256 / 8 |
| Routed / shared intermediate width | — | 512 / 512 |
| Embedding/output matrix rows | 248320 | 248320 |
| Position capacity | 262144 | 262144 |

Both use `rms_norm_eps=1e-6`, `rope_theta=1e7`, `partial_rotary_factor=0.25` and
`mrope_section=[11,11,10]`. Their full-attention layers occur at indices `3,7,...`; that pattern is
stored in config. Both have untied embedding and output matrices. Qwen3.6-27B and Qwen3.8-27B share
this Dense geometry, while training results, representations and frontend semantics can differ.

Config parsing accepts structural variation. Actual execution remains bounded by implemented
Ops: the current fused input projections have explicit Dense/MoE shapes, GDN uses the implemented
128-dimensional recurrence, and the native sparse-MoE path uses 256 experts with top-8 routing.
The current multimodal RoPE entry requires the implemented interleaved axis mapping. These checks
belong to the relevant consumers; parsing a config is not a promise to execute every size.

### Static YaRN extension

`--yarn-factor F` (`EngineOptions::yarn_factor`) selects Qwen's static YaRN extension at startup.
The artifact keeps its native `rope_parameters`, so one file serves both modes and the default route
is unchanged; the converter rejects a YaRN source config instead of storing it. With F in (1,4],
the Text RoPE of every full-attention and MTP layer uses the YaRN frequencies and magnitude of
[`ops::rope`](../../include/ninfer/ops/rope.h) with the artifact's `max_position_embeddings` as the
original length L, `beta_fast=32`, `beta_slow=1` and attention factor `0.1*ln(F)+1`, the parameters
of the Qwen3.5-family model cards and of the vLLM, SGLang and Transformers implementations. For
R=64, theta=1e7 and L=262,144, pairs 0..14 keep their frequency, pairs 22..31 divide it by F and
pairs 15..21 blend linearly; the 64 rotated dimensions are scaled by the attention factor and the
other 192 are not. The position capacity, and thus the `max_context` ceiling, becomes
`floor(F*L)`: 1,048,576 at F=4.

Scaling applies to every position, short prompts included; Qwen's cards note that static YaRN may
lower short-text quality and recommend the smallest factor that covers the workload. The YaRN Op
forms the phase in FP64, so angles stay accurate at 1M positions, where FP32 phase products would
be off by up to about 0.03 rad. MTP shares the Text RoPE configuration and is scaled with it.
DFlash and DFlash2 drafts keep their own unscaled RoPE and position capacity, so startup rejects a
`max_context` above the draft's capacity; MTP or no speculation serves longer contexts.
Measurements are in the [performance record](../performance.md#yarn-long-context-qwen38-27b).

## Logical parameters and physical bindings

[`load/`](../../src/models/qwen3_5/load/) expands the selected config into parameter names, shapes
and mathematical input Uses. The converter's
[`qwen3_5.py`](../../tools/convert/qwen3_5.py) maps source tensors into the same roles.

The main namespaces are `text/`, `vision/`, `mtp/`, `dflash/`, `dflash2/` and the optional
`proposal/`. Text block `i` contains `text/layers/i/attention/...` or `.../gdn/...`, plus its Dense
`mlp/...` or `moe/...` parameters. A projection such as `attention/query` has logical shape
`[num_attention_heads * head_dim, H]` and a Use at `text/layers/i/mixer_input`.

Bindings describe the logical values in physical parents. They can use complete objects,
consecutive regions or several ordered parts. The source `q_proj` stores each head's Q rows next
to that head's gate rows; source mapping extracts those rows into separate logical projections.
A simple split at the middle of the source matrix would change the model.

The Dense groupwise attention recipe packs Q/K into Q4 and gate/V into Q5. A supported FP8 or NVFP4
recipe can pack their union into one parent. The same four logical roles then become either the
two-weight or one-weight native fused call. Dense gate/up and MoE expert banks follow their finite
packing and execution rules. [Storage layouts](storage-layouts.md#11-logical-views-and-native-operands)
describes the difference between logical views and native operand restrictions.

`tie_word_embeddings` records a training relationship; bindings choose whether physical data is
shared. MTP and draft components reference Text's embedding/head rather than introduce private
copies by default. Shared weights keep independent Uses and activation permissions. The stored
NVFP4 weight divisor belongs to its parent; an activation divisor belongs to a mathematical input.

## Text block mathematics

In the formulas below, activations are written with feature dimensions first and token columns
last. H is hidden width, D is an attention head dimension, and T is the current token extent.
The two Text norm forms are:

```text
embedding / Vision features
  → 多层 pre-norm residual：GQA 或 GDN → Dense FFN 或 MoE
  → final norm → Main head
```

Config 决定层序、head、宽度和组件关系，原生代码决定固定数学。Text 使用 offset RMSNorm，
gain 为 `1+w`；不能在转换时先把它舍入成普通 BF16 gain。完整 GQA 带输出 gate，
Q/K 使用 head norm 和部分 interleaved MRoPE。GDN 使用因果卷积、Q/K normalization、
gated delta recurrence 和 gated norm，状态按层/head 独立。

Dense 使用 SwiGLU；MoE 使用全局 top-k softmax routed experts 与独立 sigmoid gate 的 shared
expert。表示选择不改变 route 或 shared 的数学位置。具体公式、alias 与舍入合同在
[Ops](../../include/ninfer/ops/)，模型只负责层组合与实际参数。

## 表示与执行

Converter 可以为不同逻辑角色选择 groupwise INT、NVFP4、BF16 等表示。
Activation permission 属于使用位置；实际 Op 负责其格式和几何支持。Norm、卷积与逻辑行序
由 converter/load 对齐，运行期不重打包。通用关系见[权重与加载](weights-and-loading.md)。

本地执行使用原生 Session 和 RankProgram。27B Dense 与 35B MoE 保持各自 FFN 组合，
MoE 可以使用 GPU resident 或 RTX 5090 的[混合执行](expert-weight-cache.md)。
Speculative backend、proposal head 和 Vision 在启动选择，未选择的组件不占用执行资源。

## State 与历史

完整续算状态包含 GDN recurrent/conv、continuation hidden 和所选 draft backend 的定长状态。
Main GQA、MTP 和需要 full context 的 DFlash 层分别有自己的 history。
DFlash local cyclic KV 属于 State；临时 query KV、verify records 与候选分布属于 frame。

Checkpoint 保留同一次真实执行的 State、history coverage 与位置身份。
Prefill/Replay 可在一个 Work 内推进多个数学或媒体段，保存与调度边界由 Session 统一决定。
存储、引用、Host 恢复与抢占复用[Context 管理](context-management.md)，不按模型另建缓存策略。

## Speculative backend

MTP 是一个私有 attention＋FFN block，conditioning 使用 Main 最终归一化 hidden，
输入为分别归一化后的 embedding 与 hidden 拼接。Main/MTP 稳定前沿为
`F / max(F-1,0)`；token、hidden 和位置按真实配对解释。媒体占位使用对应的 composed embedding。

DFlash/DFlash2 从指定 Main 层的 residual 捕获 conditioning，一次 masked block 提出多个 token。
DFlash2 还有动态卷积与条件候选路径。它们的 context append、proposal 分布与状态恢复和 MTP
不同，详见[Speculative 执行](speculative-decoding.md)。

Indexed proposal head 是独立可选表示，通过 row map 回到真实 token ID；target 仍使用完整
Main head。不同 proposal 只改变接受行为和成本，不改变 target 验证职责。

## Vision 与 Frontend

Qwen Vision 使用 patch projection、带位置编码的 transformer tower 和空间 merger。
其 LayerNorm、Q/K RoPE 与 GELU 形式由 Vision 数学维护，不套用 Text 的 offset norm。
共享 Qwen processor 产生 patches、媒体 spans 和三轴位置；Main 在 embedding 后替换媒体列。

Text 位置随媒体 grid 延伸，decode 保留相应位置偏移。媒体内容和位置共同参与前缀身份，
仅比较占位 token 不足以复用。预处理和 GPU 窗口采用[公共有界媒体机制](frontend.md#有界媒体准备)。

### Sparse MoE

For token h, with E routed experts and K selected experts:

```text
router_probs = softmax(W_router h)
(values, ids) = topk(router_probs, K)
routing_weight = values / sum(values)

y_e = W_down[e](SiLU(W_gate[e] h) * (W_up[e] h))
y_routed = sum(routing_weight[e] * y_e for e in ids)
shared = W_shared_down(SiLU(W_shared_gate h) * (W_shared_up h))
y = y_routed + sigmoid(W_shared_score h) * shared
```

The shared expert always executes and is outside top-k. Logical expert id e identifies router row e
and the corresponding expert matrices. Physical bank ordering and fused dispatch preserve that
relationship. Router loss coefficients and training-only outputs add no inference term.

## Prefill, decode and MTP

Text prefill gathers embedding columns, replaces media placeholders with Vision outputs, and runs
the configured blocks in chunks. It preserves state and positions across chunks. Final offset
RMSNorm and the full output head produce the target distribution. Ordinary decode repeats this
schedule for one new token. CUDA Graph replay uses the same selected model implementation.

A prefill Work retains its nominal token budget across internal rewrite and Vision
segments. RankProgram advances their state and history in order within the same frame; these
mathematical boundaries do not yield to the Scheduler. Session ends the Work at the nominal
chunk limit, a reserved capture, or the prompt frontier. Only the final prompt segment samples.

MTP is a one-layer predictor conditioned on a final-normalized target hidden state h_t and the
following token x_(t+1):

```text
e = offset_rmsnorm(composed_embedding(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t, pre_fc_norm_hidden)
u = input_projection(concat(e, h))
u = one_gated_attention_block_with_target_ffn(u)
draft_hidden = offset_rmsnorm(u, mtp_final_norm)
draft_logits = target_output_head(draft_hidden)
```

The stem concatenates embedding first, hidden second. MTP has private stem, block, final norm and
KV state. Its attention and Dense/MoE geometry come from Text. It can recursively propose several
tokens using the previous MTP hidden output.

For prompt tokens x_0 through x_n, MTP history construction uses:

```text
token inputs    = [x_1, x_2, ..., x_n]
hidden inputs   = [h_0, h_1, ..., h_(n-1)]
position inputs = [p_0, p_1, ..., p_(n-1)]
```

Tokens shift; hidden states and positions do not. At a stable target frontier F, MTP history has
frontier `max(F-1,0)`. The current round's first draft step pairs the unprocessed anchor with the
saved target hidden at F-1; no draft is carried from the preceding round. Shifted media placeholders
use their corresponding composed Vision embedding columns.
[`program/mtp.cpp`](../../src/models/qwen3_5/program/mtp.cpp) implements history alignment and
draft execution; Session owns the committed frontiers.
For speculative commit, RankProgram prepares a history-rebuild Graph for each exact batch and
accepted input width. It consumes the chosen lengths from the fixed commit frame; prefill and
ordinary alignment retain their native execution path.

An optional proposal head supplies an indexed vocabulary subset for draft prediction. Its row map
converts proposal rows to actual token IDs. Full target verification continues to use the full
output head. Backend selection, draft width and proposal-head choice are fixed at startup.

## Vision and multimodal positions

The current native processor uses 16×16 spatial patches, pairs of frames, and 2×2 spatial merge.
It consumes bytes already acquired by the CLI or server, decodes images/video frames, resizes to
merge-aligned dimensions, normalizes RGB and packs BF16 patch rows. Each row contains
`3 * 2 * 16 * 16 = 1536` values in the defined channel/temporal/spatial order. Images repeat their
frame to supply the temporal pair.

The official Vision towers have depth 27, hidden width 1152, intermediate width 4304 and 16 heads
of dimension 72. The learned position table is a 48×48 grid. Patch projection and interpolated
position addition precede the transformer stack. Each block performs:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_rope(q, k, two_dimensional_patch_positions)
a = segmented_attention(q, k, v)
x = x + output_projection(a) + bias
h = LayerNorm(x)
x = x + fc2(GELU_tanh(fc1(h) + bias)) + bias
```

Attention segments separate media/frame grids. The merger normalizes, groups 2×2 patches into
width 4608, and computes `fc2(GELU_exact(fc1(merged) + bias)) + bias`, producing Text-width columns.
Those columns replace matching placeholder embeddings. Vision has no autoregressive state.

Frontend admission uses aggregate media bytes, pixels, patch/token budgets and Engine context
capacity. The aggregate limit is 131,072 raw patches / 32,768 merged tokens; a single tower item
uses at most 16,384 merged tokens, additionally bounded by context. RankProgram executes items
sequentially and reuses their output handoff after scattering the previous item.

The prepared prompt stores axis-major `[3,T]` positions in temporal, height, width order:

- Text tokens advance all three axes together.
- Media placeholders use their merged grid coordinates; following text resumes after the maximum
  multimodal coordinate.
- Video timestamps are rendered into prompt text. Position construction uses the prepared media
  grids and token runs.
- `rope_delta = next_multimodal_position - token_count` supplies the offset for subsequent decode.

For the official Text geometry, 64 of each 256-dimensional Q/K head rotate, using 32 frequency
pairs assigned to interleaved sections `[11,11,10]`. Text-only positions reduce to ordinary partial
RoPE because the three axes are equal.

## Vocabulary and frontend resources

Keep three row domains distinct: config `vocab_size` is the embedding/output matrix extent; the
tokenizer defines addressable token IDs; an optional proposal map defines its own shortlist rows.
The official tokenizer has 248,077 addressable IDs within 248,320 matrix rows. Remaining matrix
rows are not assumed zero and do not become public tokens.

Token spelling, special-token roles and EOS are derived from loaded resources. The frontend checks
the tokenizer's supported BPE/normalization semantics and token-domain agreement. The Frontend
compiles the artifact or startup-selected Jinja template and derives exact reuse boundaries from
its rendered token sequence. Mode sampling
presets are architecture-owned, with explicit application/request overrides. Resource loading
does not infer execution identity from tokenizer filenames, release names or sampling values.

## State and numerical boundaries

Each StateImage contains the Text GDN state and continuation hidden. Per GDN layer, recurrent
state has one `[Dv,Dk]` matrix per value head, and convolution history has
`(2 * key_width + value_width) * (kernel_width - 1)` values. Startup plans Device/Host StateImage
capacity under the [context resource contract](resource-scheduling-and-context-cache.md#capacity).
ReplaySSM records are separate pending-round scratch when speculation is enabled.

Text GQA KV grows with visible context. Selected MTP owns separate KV; DFlash backends own the
state described in their reference. Session owns continuation metadata, prefix checkpoints and
directory references; RankProgram owns hidden-value and KV backing. Workspace, storage capacities
and graph buffers are derived from actual parameters and startup options by the planners described
in [Engine architecture](engine-architecture.md).

Speculation verifies the candidate sequence against processed target distributions, commits the
accepted prefix and continuation state, then publishes output. Greedy acceptance compares a
proposal token with the target argmax for that verify column. Sampling uses the defined proposal
acceptance and correction distribution. This does not impose token or logits equality between
different quantization, prefill or kernel paths.

Each floating-point Op is qualified against its independent mathematical oracle over represented
public inputs. Packed weights are decoded with their stored scales. Exact transforms use exact
oracles; private accumulation, staging and reduction order use the Op's numerical criteria.
[Op development](op-development.md) owns that qualification contract.
