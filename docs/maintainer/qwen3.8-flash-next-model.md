# Qwen3.8-Flash-Next model semantics

This reference defines the exact Text mathematics, persistent state, PLE indexing, MTP, and Vision
topology for NInfer's `qwen3.8-flash-next/mixed-nvfp4-fp8-ple-int4` target. The checkpoint is an
early preview of the Qwen4 architecture and uses the Transformers model type `qwen4_exp`; it is not
a Qwen3.6-family Variant.

The mathematical oracle is the Qwen/Hugging Face implementation generated from
[`modular_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modular_qwen4_exp.py)
and its generated
[`modeling_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
evaluated with the revision-pinned source `config.json`. NInfer may fuse operations and choose
different internal arithmetic, but each execution leaf is qualified against these represented
semantics.

## 1. Fixed dimensions and topology

| Fact | Value |
|---|---:|
| vocabulary / Text width | 248320 / 2560 |
| hyper-connection streams / concatenated width | 4 / 10240 |
| Text layers | 48 |
| full-attention layers | `3,7,11,...,47` (12) |
| Gated DeltaNet layers | all other layers (36) |
| routed experts / selected experts | 512 / 10 |
| routed and shared intermediate width | 640 |
| full-attention query / KV heads / head width | 24 / 2 / 256 |
| rotary fraction / rotary width | 0.25 / 64 |
| indexer query / KV heads / head width | 4 / 1 / 128 |
| indexer block width / token budget | 4 / 2048 |
| GDN key heads x width | 16 x 128 = 2048 |
| GDN value heads x width | 48 x 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| maximum positions / RoPE theta | 262144 / 10000000 |
| PLE injection layer | decoder layer 1 (the second layer) |
| PLE n-gram order / heads per order | bigram and trigram / 8 |
| PLE row width / logical rows | 160 / 320001536 stored rows |
| MTP | one full-attention MoE layer |
| Vision depth / width / intermediate / heads | 27 / 1152 / 4304 / 16 |

Text embeddings are repeated four times to form the initial `[4,2560]` hyper state. After the 48
decoder layers, the final hyper mixer reduces those streams to one 2560-wide state before the
independent output head.

## 2. Normalization and hyper-connections

The Text RMSNorm is one-centered. For each group `x` of the configured group size,

```text
rms(x) = x / sqrt(mean(x^2) + 1e-6)
norm(x,w) = rms(x) * (1 + w)
```

For a concatenated four-stream state `H`, the gated residual computes:

```text
N = group_rmsnorm(H), with four independent groups of width 2560
M = sigmoid(W_up(silu(W_down(N) / 4))) reshaped to [4,2560]
block_input = mean_stream(M * reshape(N,[4,2560]))
injection_weight = 2 * sigmoid(W_inject(N) / 4)
H' = H + flatten(block_output[:,None,:] * injection_weight[:,:,None])
```

Attention and MoE each own one such transition. The final and MTP input mixers omit
`W_inject` and return only `block_input`.

## 3. Gated DeltaNet layers

The input projection produces `[q,k,v,z] = [2048,2048,6144,6144]`; a depthwise causal
four-tap convolution followed by SiLU applies to `[q,k,v]`. The 16 key heads are repeated three
times to align with 48 value heads. `q` and `k` are independently L2-normalized. Per value head,

```text
beta = sigmoid(Bx)
g = -exp(A_log) * softplus(Ax + dt_bias)
S_t = exp(g_t) * S_(t-1)
delta_t = (v_t - k_t^T S_t) * beta_t
S_t = S_t + k_t delta_t^T
o_t = q_t^T S_t
```

The recurrent matrix state is FP32 `[48,128,128]` per sequence and layer. The convolution state is
the previous three `[q,k,v]` values `[10240,3]`. The output applies RMSNorm to each 128-wide value
head, multiplies by its direct ones-initialized gated-norm weight (this weight does not use the
ordinary Text norm's unit offset), multiplies by `silu(z)`, flattens to 6144, and projects to 2560.
Prefill may use an equivalent chunked delta-rule evaluation; decode must preserve the same state
transition.

## 4. QSA full-attention layers

The main projection stores each query head as `[query_256,output_gate_256]`, followed by 512 key
rows and 512 value rows. Query and key heads use one-centered RMSNorm and 64 rotary dimensions.
After causal sparse attention, each query head output is multiplied by `sigmoid(output_gate)` before
the 6144-to-2560 output projection.

The QSA indexer projects four 128-wide queries and one raw 128-wide key per token. For every visible
prefix, it partitions the complete visible tokens into consecutive four-token blocks and leaves an
incomplete tail uncompressed. Each complete block key is the FP32 mean of its four raw keys,
one-centered RMS-normalized, and rotated at the block's first position. The score for a block is:

```text
score = sum_head(relu(q_head dot pooled_block_key)) / sqrt(128)
```

The indexer selects up to `2048 / 4 = 512` blocks, expands them back to tokens, appends the visible
tail, and intersects that mask with the causal mask. Raw indexer keys, main keys, main values, and
full MRoPE positions are persistent per-sequence state.

## 5. Mixture of experts

The router computes a 512-way FP32 softmax, selects the top 10 probabilities, and does not
renormalize them (`norm_topk_prob=false`). For selected expert `e`,

```text
expert_e(x) = down_e(silu(gate_e(x)) * up_e(x))
routed(x) = sum_selected(prob_e * expert_e(x))
shared(x) = sigmoid(shared_gate(x)) * shared_down(silu(shared_gate_proj(x)) * shared_up(x))
moe(x) = routed(x) + shared(x)
```

The production expert-bank kernel may group tokens and use native NVFP4 W4A4, but expert selection,
probability multiplication, and the FP32 softmax remain part of the semantic contract.

## 6. Per-Layer Embedding

PLE has eight bigram and eight trigram heads. The table is 128 physical shards of 2500012 rows;
each logical head selects a global row from its own prime-sized subrange. Token multiplication and
XOR use wrapping signed int64 semantics. For current token `t`, previous segment tokens `p1,p2`,
stored multipliers `m0,m1,m2`, prime vocabulary `v_h`, and offset `o_h`:

```text
b = wrap64(t*m0) XOR wrap64(p1*m1)
g = b XOR wrap64(p2*m2)
row_h = o_h + floor_mod(h < 8 ? b : g, v_h)
```

History begins with `[eos,eos]`; committing EOS resets both history positions. PLE lookup therefore
participates in speculative accept/rollback rather than mutating history during proposal.

The 16 selected 160-wide rows concatenate to a 2560-wide embedding `E`. The layer computes a
10240-wide key, a 2560-wide value, and normalized four-stream query. Per stream:

```text
raw_gate = dot(norm(key_stream), norm(query_stream)) / sqrt(2560)
gate = sign(raw_gate) * sqrt(max(abs(raw_gate),1e-6))
V_stream = sigmoid(gate) * value(E)
PLE(H) = V + silu(depthwise_dilated_conv(group_rmsnorm(V)))
```

The convolution has four taps and dilation 3, so its persistent history length is nine. PLE is added
to the four-stream hidden state before layer 1's attention hyper-connection.

## 7. MTP and Vision

MTP normalizes and independently projects the current token embedding and the chosen main hidden
state, adds them, repeats/mixes the four hyper streams, then executes one full-attention QSA + MoE
layer with its own weights. Proposal and commit must transactionally include main KV, indexer keys,
GDN state, PLE token/convolution history, and MTP state.

Vision is the checkpoint's 27-block 1152-wide tower: 3D patch projection, learned position
embedding, non-causal 16-head attention, GELU MLP, and a 2x2 patch merger from 4608 to Text width
2560. Text uses interleaved three-axis MRoPE with sections `[11,11,10]`; the QSA indexer additionally
retains full positions for cached raw keys.
