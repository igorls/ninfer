# Qwen4Exp 架构

Qwen3.8-Flash-Next 使用 Qwen4Exp：GDN/QSA、四路 Gated Residual、MoE 与 PLE 的组合。
实例维度来自 config，逻辑权重来自绑定；不根据 checkpoint 名称选择另一套层实现。
主要入口为 [config](../../src/models/qwen4_exp/config.cpp)、
[load](../../src/models/qwen4_exp/load/)和 [execution](../../src/models/qwen4_exp/execution/)。

## 主干与状态

```text
embedding / Vision features → 四路 residual
  → attention HC → GDN 或 QSA → HC combine
  → FFN HC → routed + shared MoE → HC combine
  → final HC mixer → Main head
```

PLE 在配置指定层的 attention HC 前加入。HC 的 streams 是特征轴，不增加 GDN、QSA 或 MoE
的 token 数。Final mixer 的 sample hidden 与 mixer 前的四流 residual 是不同结果，
MTP conditioning 使用后者。

GDN 从 HC 输入形成 QKVZ 和 controls。Q/K/V 经过因果卷积与 SiLU，Z 绕过卷积；
递推输出经过直接 gamma 的 head norm 和 `sigmoid(Z)`，再投影回 hidden。
Conv 的 SiLU 与输出 gate 的 sigmoid 分属不同位置。

QSA 保存完整主 K/V；四 token 压缩只作用于 index key。完整块的 pooled key 经 norm/RoPE
参与打分，所有 query heads 共用 TopK 块集合；Sparse GQA 另消费未完成块的可见 tail。
Indexer score 不加入 attention logits。Raw index tail 与位置属于定长 State，完整历史属于 KV。

完整 checkpoint 还包含 GDN、PLE conv、continuation hidden 及启用 MTP 时所需的状态。
Raw token 账本与模型前沿归 Session。具体恢复、共享和分配沿用[Context 管理](context-management.md)。

## 权重选择与合并

Flash Next recipe 采用以下主要取舍；精确参数映射由
[converter](../../tools/convert/qwen4_exp.py)维护。

| 角色 | 选择 |
|---|---|
| Main routed experts | 导入 NVIDIA NVFP4，允许 A4 |
| GDN 大投影、Main QSA Q/gate 与 O | Q8 A16 |
| MTP routed experts | Block-FP8，允许 A8 |
| HC、控制投影、shared、主 head、Vision、其他保留投影 | BF16，保留对应数学边界 |
| PLE table | NF4，精确 BF16 lookup 出口 |
| Optimized MTP head | 从 Main 源 head 额外生成完整词表 Q4 A16，不裁词表 |

Offset norm 保留源 offset words，`1+w` 与 HC 缩放不提前折入低精度权重。
GDN/PLE convolution 转为 tap 外、channel 连续的布局，保持 tap 时间次序。
NVFP4 weight divisor 与 activation divisor 分属权重和 Use，见[权重与加载](weights-and-loading.md)。

合并投影保留逻辑角色。Main QSA 的不同表示分开；MTP 的 K、V、index-K 放在同一个 BF16 parent
的连续前缀，随后才是 Q、gate、index-Q。完整投影与 history-only 计算共享权重，避免补齐历史时
多次小 Linear。TP 先按角色切片再合并，不能把整个 QKV 类 parent 简单对半截断。
LinearSplit 直接写各段连续输出，模型不重复整理这些投影结果。

## PLE

PLE 根据 raw token 的二、三元组 hash 查表，组合 lookup embedding，再经投影、gate 与
带 dilation 的卷积作用于 residual。视觉特征不替换 hash 输入的 token IDs。
Lookup、投影和卷积有各自表示与历史合同，表的驻留位置不改变 Main 的消费语义。

磁盘路径从 V3 NF4 表按行读取，使用有界行缓存和准备/上传槽；输入已知部分可以增量准备，
计算消费稳定的 embedding 与就绪依赖。上传槽直到最后 GPU 消费者结束才复用。

双机 Device 路径按表 shards 放置，各 rank lookup 后合并精确 words，再执行复制的 PLE 计算。
数据生产可与前序计算重叠；Main 抓图不按磁盘或 Device 来源任意切碎。
PLE 缓存不属于请求 State，只有卷积历史和真实 token 前沿参与恢复。

实现入口：[PLE](../../src/models/qwen4_exp/ple/)、
[Device lookup 组合](../../src/models/qwen4_exp/program/device_ple.cpp)。

## 并行与放置

| 部分 | 双 Spark 分工 |
|---|---|
| GDN / Sparse GQA | 按完整 heads TP；O 产生局部贡献后归约 |
| QSA Indexer | 投影、评分、index cache/tail 复制，供本地 GQA 使用 |
| HC / residual | 复制；消费完整归约结果 |
| Routed experts | TP2/EP1 时每个 expert 的 intermediate TP；TP2/EP2 时完整 expert EP |
| Router / shared | Router 复制，shared intermediate TP；贡献统一汇合 |
| Embedding / head | Embedding 复制，head 按词表 TP |
| PLE / Vision | PLE 表分片；Vision 权重复制、独立编码单元分工 |

EP 保留全局 expert IDs 和 route 系数，不在本地集合重新归一化。TP 的 GU 与 down 沿对应
intermediate 同时切分，block-FP8 必要时保值 refine。共享专家只贡献一次完整数学结果。

Router/Indexer 的复制决策使用一致的输入与计算条件；模型需要共同的离散结果，不能用单个
Op 的近似精度容限替代一致性要求。具体切片、expert 映射与通信由
[ParallelPlan](../../src/models/qwen4_exp/parallel/plan.cpp)维护。

RTX 5090 还可通过[expert cache](expert-weight-cache.md)执行 Main experts 的 CPU/GPU 混合路径。
Spark 使用 GPU 计算；Context 每 rank 持有本地 State/KV，普通 TP histories 构成完整共同恢复点。

## MTP、Vision 与 Graph

MTP 使用私有 HC/QSA/MoE 层，与 Main 共享 embedding，默认共享 Main head；optimized 时只改变
其 draft head 绑定。Conditioning 为 Main 四流 residual，stem 保持这一特征关系。
稳定前沿、首步采用和真实 conditioning 的 history repair 见[Speculative 执行](speculative-decoding.md)。

Vision 复用 Qwen tower 与 MRoPE 输入机制。编码 features 与 prefill 窗口按真实媒体 spans 交接，
状态与前缀身份同时包含媒体和位置。资源机制见[Frontend](frontend.md)。

Hidden norm 沿完整 10240 维归约，随后同一个 fc_hidden 独立作用于四条 streams。
第一步使用 Main conditioning，后续 draft step 使用前一步 MTP 的 pre-final-mixer
residual。MTP 的 QSA/MoE 层及 final HC mixer 产生下一组 conditioning 和 sample。
Draft 默认使用 Main head，也可选择第 3.3 节的独立量化 head；target 仍使用 Main head。

Vision 将 Conv3d patch 按 C/T/H/W 展平为 1536 维。Blocks 使用非因果 attention、
LayerNorm 和 `gelu_pytorch_tanh` MLP；merger 在 2×2 合并前做 LayerNorm，
两次 FC 之间使用 GELU。本实例没有 deepstack。编码器复用
[公共 Qwen Vision](../../src/models/common/qwen/vision/)，Text 层只消费其 2560 维输出。

## 3. Flash Next 量化 recipe

The fork additionally carries `qwen3_8_flash_next_mixed`: official BF16 source to
canonical NVFP4 Main/MTP expert banks, row-FP32 FP8 Text mixer projections, U4Z8
PLE, and BF16 elsewhere. Each NVFP4 expert retains its own weight divisor;
each A4 Use stores an explicit unit global activation divisor. Native preparation
chooses supported A16/A8/A4 routes from the stored operands and Use permissions.
The narrow K/V tail of the fork's FP8 QSA parent uses A16 activations.
Published fork packed-v3 files require the explicit offline
[reframe](../weight-conversion.md#qwen4-family-qwen4_exp-from-official-safetensors).
The table below describes the separate upstream NVIDIA-backed recipe. These
representations and their quality/performance qualification must remain distinct;
see [upstream ports](upstream-ports.md#final-tip-adaptation-in-progress-october-22).

`qwen3_8_flash_next` 使用 BF16 主 checkpoint 提供 config、资源、Main 非 routed
权重、Vision 和 PLE；`--source nvidia=PATH` 提供完整
`nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint，导入 Main routed experts 和 MTP 独有参数。
来源和转换方法写入 `.conversion.json`。完整选择由
[official recipe](../../tools/convert/official_recipes.py)维护。

| 部分 | 持久表示 | 计算许可或消费 |
|---|---|---|
| Main routed gate/up/down | NVIDIA NVFP4，G16 E4M3FN scale，每 expert 矩阵一个 FP32 divisor | `AllowA4` |
| Main GDN QKVZ/output、QSA Q/gate/output | BF16 源转 Q8 G32/FP16 scale | `A16Only` |
| HC、QSA K/V/indexer、PLE 投影、router/shared FFN | 原 BF16 words | `A16Only` |
| GDN AB、convolution、norm，其他 norm/bias | 原 BF16 words | 各 Op 数学 |
| GDN A_log / dt_bias | 源 BF16 精确展开为 FP32，各 `[48]` | 保留原始含义 |
| Token embedding / Main LM head | 两份独立 BF16 对象 | lookup / A16 head；MTP 默认共享 |
| Optimized MTP head，可选 | BF16 Main head 源转 Q4 G64/FP16 scale，完整词表 | `A16Only` |
| PLE table | BF16 源转 `nf4_g16_e4m3fn` | 精确解码为 BF16 lookup 输出 |
| MTP routed experts | NVIDIA E4M3FN codes，128×128 块 BF16 multiplier | `AllowA8` |
| MTP 其他独有参数、全部 Vision | 原 BF16 words | 投影 `A16Only` |
| PLE hash controls | 原始 35 个 INT64 | 无损 Resource |

GPU MoE 已有 NVFP4 A4、block-FP8 A8/G128 及小批量 A16 路径；
各语义入口的私有 shape policy 选择实际计算。CPU Main experts 使用 BF16 输入和
FP32 中间计算，不量化激活。许可允许高精度路径，并不要求每次调用都量化为最低位宽。

Q8 使用 `grouped_absmax`，沿 K 每 32 个值分组。合并和行置换先确定最终行序，
再进行编码。Packing、scale swizzle 和 convolution 转置在转换时完成，loader 绑定原字节。
KV、recurrent state 和 workspace 的 dtype 独立于权重 recipe。

### 3.1 Norm 参数

| 角色 | 持久参数 | 归约与 affine |
|---|---|---|
| HC、PLE grouped RMSNorm | BF16 offset `[10240]`，native `[2560,4]` | 各 stream 独立归约，gain=`1+w` |
| Main/MTP QSA Q/K、indexer Q/K | BF16 offset `[256]` / `[128]` | 每 head 归约，同角色 heads 共享参数 |
| MTP pre-FC embedding / hidden | BF16 offset `[2560]` / `[10240]` | 对整个对应向量归约 |
| GDN output gated RMSNorm | BF16 gamma `[128]` | 直接 gain，所有 V heads 共享 |
| Vision LayerNorm | BF16 weight/bias `[1152]` | 直接 affine；merger norm 位于 merge 前 |

Offset 保留源 words，`1+w` 不预先舍入为 BF16 gain；HC 的 `/4` 不折入权重。
参数 dtype 与归约精度是不同的约定。

### 3.2 NVIDIA 编码导入

Main 源的 `weight_scale_2`、`input_scale` 是正有限 FP32 multipliers。
[ModelOpt adapter](../../tools/convert/sources/modelopt_nvfp4.py)使用：

```text
source weight = E2M1(code) * E4M3FN(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
stored weight = E2M1(code) * E4M3FN(scale) / d_w
```

Codes/scales words 保留后按目标 layout 重排；FP32 reciprocal 舍入是元数据变换，
不承诺所有解码值与源逐位相同。Recipe 要求同 expert 的 gate/up 两个 multiplier
words 相等，并要求一层所有 experts 的 GU 输入 multiplier 相等、down 输入 multiplier
相等。每个 expert 仍各自持有 weight divisor。

Activation divisors 存为 FP32 `[48,2]`，元素 `2*l` 与 `2*l+1` 分别供 GU/down Use
引用；它们与 weight divisor 分开解释。A4 输入按 G16 量化，product 从 FP32 直接编码，
不增加 BF16 中间 seam。

MTP `fp8_e4m3fn_block128x128_bf16` 重建为 E4M3FN code × BF16 block scale。
Gate/up 各自保留原 scale，640 行边界与 128 行块对齐。A8 按每 token 的 K128 分组
动态量化，scale 为 FP32；每个 K128 dot 分别缩放后累加，没有静态 activation divisor。

### 3.3 Optimized MTP head

转换时 `--proposal` 从 Main BF16 head 源生成独立 `proposal/head`，使用
`grouped_absmax`、`q4_g64_fp16` 和 `row_split_k128_v1`。它保留全部
`[248320,2560]` 物理行及原 vocabulary ID 顺序，不裁剪词表、不创建 `proposal/token_ids`；
目录声明 `text.proposal.domain="full"`。`mtp/final_hidden` Use 的 policy 为 `A16Only`。

Main BF16 head 和 `mtp/output_head` 的共享绑定保留。启动选择 optimized 时，
MTP 改为绑定 `proposal/head`，target 验收和 ordinary decode 继续使用 Main head。
两份 head 采用相同的词表 TP 分区：TP2 下每 rank 各持有 `[124160,2560]` 的本地部分，
沿用全局 token ID 和候选汇合，不需要 token remap。

这是附加的 draft 表示；关闭 optimized head 时不加载其对象。
启动选项和执行流程见[执行与状态](qwen4_exp-execution.md#5-mtp-的生成与采用)。

## 4. 合并矩阵与持久布局

逻辑矩阵写作 `W[N,K]`；N 是输出行，K 是输入特征，区间均为半开区间。
相同输入、表示和兼容元数据的投影沿 N 合并，逻辑角色仍由各自 Binding 引用。
Native Tensor 第 0 维连续；`M=T*B`，`column=t+T*b`。

| Parent | Main / MTP 数量 | 逻辑 shape | 格式 | 输出行顺序 |
|---|---:|---|---|---|
| GDN QKVZ | 36 / 0 | `[16384,2560]` | Q8 | Q 2048、K 2048、V 6144、Z 6144 |
| GDN AB | 36 / 0 | `[96,2560]` | BF16 | a 48、b 48 |
| GDN output | 36 / 0 | `[2560,6144]` | Q8 | 原行序 |
| Main QSA Q/gate | 12 / 0 | `[12288,2560]` | Q8 | Q 6144、gate 6144 |
| Main QSA KV/index | 12 / 0 | `[1664,2560]` | BF16 | K 512、V 512、index-Q 512、index-K 128 |
| Main QSA output | 12 / 0 | `[2560,6144]` | Q8 | 原行序 |
| MTP QSA input | 0 / 1 | `[13952,2560]` | BF16 | K 512、V 512、index-K 128、Q 6144、gate 6144、index-Q 512 |
| MTP QSA output | 0 / 1 | `[2560,6144]` | BF16 | 原行序 |
| Block HC down/inject | 96 / 2 | `[324,10240]` | BF16 | down 320、injection 4 |
| Final HC down | 1 / 1 | `[320,10240]` | BF16 | 全部 down |
| HC up | 97 / 3 | `[10240,320]` | BF16 | stream-major，`row=s*2560+d` |
| PLE KV | 1 / 0 | `[12800,2560]` | BF16 | 四流 key 10240、共享 value 2560 |
| MoE input | 48 / 1 | `[1793,2560]` | BF16 | router 512、shared score 1、shared gate 640、shared up 640 |
| Shared down | 48 / 1 | `[2560,640]` | BF16 | 原行序 |
| Routed GU bank | 48 / 1 | `[512,1280,2560]` | Main NVFP4 / MTP block-FP8 | expert-major，每 expert 内 gate 640、up 640 |
| Routed down bank | 48 / 1 | `[512,2560,640]` | Main NVFP4 / MTP block-FP8 | expert-major |
| MTP fc_embedding / fc_hidden | 0 / 各 1 | `[2560,2560]` | BF16 | 独立权重；分别处理 M / 4M 列 |
| Main LM head | 一份，MTP 默认共享 | `[248320,2560]` | BF16 | vocabulary 顺序 |
| Optimized MTP head | 0 / 可选 1 | `[248320,2560]` | Q4 | 完整 vocabulary 顺序 |

Main QSA 的两个 input parents 精度不同；MTP 六段投影合为一个 BF16 parent。
MTP 的 K/V/index-K 位于前 1152 行，可直接形成 `[1152,2560]` 的连续权重 view；
完整投影与三段投影共享同一份权重。TP2 按相同角色顺序组装本地 parent：
K 256、V 256、index-K 128、Q 3072、gate 3072、index-Q 512，
完整矩阵为 `[7296,2560]`，连续前缀为 `[640,2560]`。
MoE control 为前 513 行，shared GU 为后 1280 行，可作零拷贝 views。
持久合并不强制一次投影调用；LinearSplit 可以直接产出独立连续的激活分段。

### 4.1 行置换与 convolution

源 q_proj 按 head 交错 Q/gate，转换时解交错：

```text
Q[h*256+d,:]    = source_q_proj[h*512+d,:]
Gate[h*256+d,:] = source_q_proj[h*512+256+d,:]
```

GDN 与 PLE 的 convolution 都转为 channel 连续：

```text
source [C,1,4] → artifact/Binding [4,C] → native Tensor [C,4]
stored[j*C+c] = source[c*4+j], C=10240
```

这两类变换均保留源 BF16 words；卷积 tap 时间顺序不变。PLE 的 dilation=3 由计算解释。
Norm、bias 与 convolution 不作为 Linear 权重重新量化。

### 4.2 Vision 与 expert banks

| Vision 参数 | 数量 | 持久 shape |
|---|---:|---|
| Patch projection | 1 | `[1152,1536]` |
| QKV / bias | 27 | `[3456,1152]` / `[3456]`，Q/K/V 依次排列 |
| Attention output | 27 | `[1152,1152]` |
| FC1 / FC2 | 各 27 | `[4304,1152]` / `[1152,4304]` |
| Merger FC1 / FC2 | 各 1 | `[4608,4608]` / `[2560,4608]` |
| Position embedding | 1 | `[2304,1152]` |

每层 routed experts 有 GU、down 两个物理 parent；逻辑 gate/up 各
`[512,640,2560]`，down 为 `[512,2560,640]`。Binding 按 expert ID 引用 GU 的两半
和 down 全矩阵。Binding range 按 C-order 逻辑元素计数，不包含 scale/divisor bytes。

Main 使用 `batched_block_scale_k16_m128x4_v1`，bank 内依次存所有 codes、
swizzled scales、FP32 divisors `[512]`。GU/down payload 分别为
943720448 / 471861248 B。Scale swizzle 使用 expert 内的局部行号。
MTP 使用 `block_scale_m128_k128_v1`，codes 后为 BF16 scales
`[512,10,20]` / `[512,20,5]`，payload 分别为 1677926400 / 838963200 B。

Recipe 按资源/辅助值、Main 固定权重、Vision、MTP、Main experts、PLE shards 放置。
Main experts 按 layer→GU/down，PLE 按 shard ID；普通对象对齐 256 B，PLE 对齐 4096 B。
对象可以跨 V3 续卷，实际文件范围由目录映射，不能从文件名推导参数身份。

## 5. PLE 表、hash 与 lookup 出口

PLE 的 128 个 NF4 parents 保留原 shard 边界和各自 divisor。
`stored_scale_mse` 对每 shard 做有界两遍转换；误差计算、scale 搜索和平局规则见
[转换数值合同](tensor-formats.md#36-stored_scale_mse-conversion-method)。

`row_paged_g16_4k_v1` 每行 90 B：80 B packed codes、10 B scales。
每个 4 KiB 数据页存 45 行，末页 37 行；每 shard payload 为 227557380 B。
全局行 g 属于 `shard=g/2500012`、`row=g%2500012`，hash head 不对应固定 shard。
具体 padding、divisor 地址和编码见[布局合同](storage-layouts.md#6-row_paged_g16_4k_v1)。

`text.resources.ple_hash_controls` 是 280 B 的 `raw_bytes_v1` Resource，顺序为：
三个 layer multipliers、十六个 head vocab sizes、十六个 head offsets，
均为源 little-endian signed INT64 words。控制值无损复制，不重新生成 seed/prime。

当前位置 raw ID 为 x0，前一、前二枚为 x1、x2；负位置补 EOS。
若 x1=EOS，仅对本次查询令 x2=EOS；x0=EOS 仍消费此前历史：

```text
u2 = word64(x0*multiplier[0]) XOR word64(x1*multiplier[1])
u3 = u2 XOR word64(x2*multiplier[2])
g[h] = positive_mod(signed64(h<8 ? u2 : u3), head_vocab_size[h])
       + head_offset[h]
```

乘法保留 modulo-2^64 words，再解释为 signed64 做正模。输出按 2-gram 八个 heads、
3-gram 八个 heads 拼接。分段与增量 append 使用真实 raw-ID tail；视觉特征不替换 IDs，
EOS 规则不清空 PLE convolution history。实现见 [hash](../../src/models/qwen4_exp/ple/hash.cpp)。

Lookup 出口严格为：

```text
LUT[shard,scale_word,code] =
    BF16_RNE(codebook[code] * E4M3FN(scale_word) / divisor[shard])
```

它对精确表示值直接做 BF16 RNE，不能用未经证明的 FP32→BF16 双舍入替代。
Scale words 的有效域为 `0x00..0x7e`；零结果保留 codebook 的符号。
每 shard 的不可变 LUT 占 4096 B，合计 512 KiB；行解码写出 160 个 BF16 words，
十六行按 head 顺序组成 `E[2560,T,B]`。通用实现见
[NF4 Host 数值辅助](../../src/core/nf4.h)，异步准备和寿命见执行文档。

## 6. 权重容量与驻留

下面是此 recipe 的物理权重及控制值，不包含 frontend resources、对象间对齐、
目录和文件 framing。它不是启动时的总内存占用。

| 部分 | Bytes | GiB | 当前放置 |
|---|---:|---:|---|
| Main 非 routed-expert、非 PLE 表权重 | 7418143232 | 6.908684 | GPU |
| Main routed experts | 67947921408 | 63.281433 | GPU resident，或 registered Host backing 加 RankProgram GPU cache |
| PLE NF4 表 | 29127344640 | 27.126954 | V3 磁盘原位 backing |
| Vision | 897862112 | 0.836199 | 启用时 GPU |
| MTP experts | 2516889600 | 2.344036 | 启用时 GPU |
| MTP 其他独有权重 | 181136896 | 0.168697 | 启用时 GPU |
| Activation divisors 与 PLE hash controls | 664 | — | 只读控制值 |
| 不含 optimized head 合计 | **108089298552** | **100.666004** | — |
| Optimized MTP head，可选 | 337715200 | 0.314522 | 启用时 GPU；TP2 每 rank 一半 |
| 含 optimized head 合计 | **108427013752** | **100.980526** | — |

Text-only 权重及控制值约 97.317072 GiB。全部组件在 offload 模式的固定 GPU 权重约
10.257616 GiB，另计 Main expert cache；resident 模式的 GPU 权重约 73.539049 GiB。
两种模式均另计 StateImage、KV、workspace、Graph 和 staging。
未启用的 Vision/MTP 不 materialize；共享 embedding/head 不重复计费。
启用 optimized head 在上述 GPU 权重之外增加约 322.07 MiB；
TP2 每 rank 增加约 161.04 MiB。

Offload 模式的 Host backing 包含已缓存到 GPU 的 experts，eviction 无需写回；resident
模式直接绑定 GPU bank，不保留 Host expert 副本。PLE 启动只读目录、控制值和 divisors，不整体加载表。
ModelDefinition 持有配置、表示描述与 Host 控制值；ModelShard 持有本地不可变权重与 PLE 文件 backing。
Session 管理逻辑状态和上下文目录，RankProgram 持有可变缓存和物理执行资源。
完整资源预算见[执行与状态](qwen4_exp-execution.md)。

## 7. 数学来源

源结构参照固定版本的
[Transformers 模型](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)、
[配置](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)及
[vLLM MTP](https://github.com/vllm-project/vllm/blob/6fddc9b21e763bfc947cc01d3ce815179aac1dc6/vllm/models/qwen4_exp/nvidia/mtp.py)。
[NVIDIA 来源说明](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fc694b54fb0174e0913e6adf86691ef85a4ead47/README.md)
记录量化来源。NInfer 的正式数值边界由生产 Op 合同维护；上游 TP、私有 casts 和
kernel packing 不自动成为本项目的数学或持久合同。
