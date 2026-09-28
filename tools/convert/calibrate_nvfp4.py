"""Measure NVFP4 activation input divisors for locally encoded Qwen3.5 dense projections.

A locally encoded NVFP4 parent that admits A4 compute needs one FP32 activation divisor per
input site. The runtime quantizes each 16-value activation block with the E4M3FN scale
``E4M3(d_x * max|block| / 6)``, saturating at 448, so a site divisor fixes the largest input
magnitude the A4 route represents without clipping:

    d_x = binary32(2688 / max|x|)        # 2688 = 6 * 448, max over the fixed corpus

This tool evaluates the official BF16 checkpoint with Hugging Face Transformers, records the
maximum absolute BF16 input of every projection site over a fixed corpus, and writes the
calibration document that ``qwen3_8_27b_nvfp4full`` consumes through
``--source calibration=PATH``. Only the sites the recipe encodes locally are written; the recipe
rejects a document whose site set differs.

The fixed corpus and the site convention are adapted from cometkim/ninfer
``tools/convert/qwen3_8_27b/calibrate_nvfp4full.py`` at 55152a4f (Apache-2.0). The model is
loaded whole (about 52 GiB of BF16 Text weights; ``device_map="auto"`` spills to host memory on
a smaller device) instead of that tool's layer streaming.

    python3 -m tools.convert.calibrate_nvfp4 --model /path/to/Qwen3.8-27B \\
        --out qwen3_8_27b_nvfp4full_calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

FULL_RANGE = 2688.0
WINDOW_TOKENS = 4096
MODEL = {"repository": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
# Pinned sources of the broad-v1 corpus, disjoint from the kld-400k-v1 evaluation slices.
BROAD = {
    "streams": (
        "eval/corpora/perplexity-1m/data/wikitext/01.txt",
        "eval/corpora/perplexity-1m/data/pg19/01.txt",
        "eval/corpora/perplexity-1m/data/zhwiki/01.txt",
        "eval/corpora/perplexity-1m/data/ninfer/01.txt",
    ),
    "stream_tokens": 16384,
    "belebele": {
        "repository": "facebook/belebele",
        "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
        "languages": (
            "ita_Latn", "nld_Latn", "pol_Latn", "tur_Latn",
            "vie_Latn", "kor_Hang", "zho_Hans", "ind_Latn",
        ),
        "tokens": 4096,
    },
    "math": {
        "repository": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "file": "test.jsonl",
        "first_row": 300,
        "tokens": 16384,
    },
    "s1k": {
        "repository": "simplescaling/s1K-1.1",
        "revision": "96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d",
        "file": "data/train-00000-of-00001.parquet",
        "rows": (20, 40),
    },
    "ultrachat": {
        "repository": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "file": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        "rows": (24, 72),
    },
}

# Fixed calibration corpus, byte-identical to cometkim/ninfer@55152a4f: prose, technical writing,
# source code, numeric tables, French, Chinese, Spanish, JSON and Markdown.
CALIBRATION_DOCUMENTS = (
    """The history of numerical weather prediction begins with Lewis Fry Richardson's 1922
attempt to compute atmospheric pressure changes by hand. His forecast, produced by a room
of human calculators working in shifts, took six weeks to advance the atmosphere by six
hours and was wildly inaccurate. Yet the method he described - dividing the atmosphere into
a grid of cells and applying the equations of fluid dynamics to each cell - is precisely
what every modern forecast model does. The difference is that where Richardson needed
weeks of human labor, a contemporary supercomputer performs the same arithmetic in a few
seconds, on a grid a hundred times finer, and assimilates observations from satellites,
aircraft, ocean buoys, and ground stations to initialize the computation. The forecasting
problem is now less about raw computation than about representing small-scale physics:
clouds, turbulence, and convection that fall below the grid resolution.""",
    """Gradient-based optimization underlies nearly all of modern machine learning. Given a
differentiable loss function L parameterized by weights w, training iterates w <- w - eta
* grad L(w), where eta is the learning rate. Stochastic gradient descent estimates the
gradient from minibatches; momentum accumulates a velocity v <- mu * v - eta * grad to
damp oscillations across ill-conditioned valleys; and adaptive methods such as Adam
maintain per-parameter running estimates of the first and second moments, rescaling the
step by an estimate of the gradient's magnitude. Careful initialization matters: drawing
weights from a distribution whose variance matches the fan-in and fan-out of each layer
keeps activation magnitudes stable as signals propagate forward and gradients propagate
backward. Normalization layers - batch norm, layer norm, RMSNorm - remove mean and scale
drift, allowing higher learning rates. Learning-rate schedules decay eta over training:
warmup ramps it up over the first few thousand steps, cosine schedules anneal it toward
zero, and step schedules divide it by a constant at fixed milestones. Regularization by
weight decay, dropout, and data augmentation counteracts overfitting.""",
    """def build_frequency_table(samples: list[int], size: int) -> list[float]:
    counts = [0] * size
    for value in samples:
        if not 0 <= value < size:
            raise ValueError(f"sample out of range: {value}")
        counts[value] += 1
    total = sum(counts)
    if total == 0:
        return [0.0] * size
    return [count / total for count in counts]

def entropy(table: list[float]) -> float:
    from math import log2
    return -sum(p * log2(p) for p in table if p > 0.0)

def rolling_checksum(data: bytes, window: int = 32, modulus: int = 16777619) -> int:
    if window <= 0:
        raise ValueError("window must be positive")
    digest = 0
    for i, byte in enumerate(data):
        digest = (digest * 31 + byte) % modulus
        if i >= window:
            digest = (digest - data[i - window] * pow(31, window, modulus)) % modulus
    return digest

class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[float] = []
        self.head = 0

    def push(self, value: float) -> float | None:
        evicted = None
        if len(self.items) == self.capacity:
            evicted = self.items[self.head]
            self.items[self.head] = value
            self.head = (self.head + 1) % self.capacity
        else:
            self.items.append(value)
        return evicted""",
    """Quarterly report extract, all figures in thousands of dollars unless noted.
Revenue: Q1 4,812; Q2 5,377; Q3 5,940; Q4 6,512; full year 22,641, up 18.4 percent
year over year. Cost of revenue: Q1 2,105; Q2 2,318; Q3 2,504; Q4 2,691. Gross margin
improved from 56.2 percent in Q1 to 58.7 percent in Q4. Operating expenses: research and
development 6,204, sales and marketing 4,481, general and administrative 1,902. Operating
income 2,363 for the year, margin 10.4 percent. Net income 1,977 after interest expense
of 214 and tax provision of 172 at an effective rate of 9.6 percent. Cash and equivalents
11,340 at year end; inventory 2,905; accounts receivable 3,661 with days sales
outstanding of 51. Capital expenditure 1,208, of which 744 related to compute capacity.
Headcount ended at 612, up from 545, of which 289 in engineering. Contracted backlog
stood at 9,880 with average duration of 2.3 years.""",
    """Le vent se levait sur la plaine de Beauce balayant les champs de blé presque mûrs.
Les paysans avaient fini la moisson dans le voisinage et les meules dorées attendaient
les chariots. Au loin, on apercevait la flèche de la cathédrale, grise sur le ciel
encore clair, tandis que des nuages épais montaient à l'horizon du côté de la forêt.
La chaleur de la journée s'attardait dans l'air immobile du soir. On entendait
seulement le froissement des feuilles sèches et, de temps en temps, l'aboiement d'un
chien dans quelque ferme éloignée. Elle marcha longtemps sur le chemin creux entre les
haies, ne pensant à rien, regardant la terre de Beauce s'endormir lentement dans la
lumière déclinante. Il lui semblait que cette plaine interminable portait tous les
travaux et toutes les saisons de sa vie, et que chaque sillon connaissait son nom.""",
    """一九四三年秋，昆明的雨季来得比往年早。联大的教室屋顶是铁皮的，雨点打上去，
声音大得讲课的人都听不见自己说话。教授停下来，学生们便合上笔记，听雨。有人
后来回忆说，那几年的知识有一半是在雨声里学的。物价一日三涨，教授们在实验室
外面种菜，在中学兼课，把藏书一册一册地卖出去。可是图书馆晚上依然坐满了人，
一盏油灯下面常常是两个人。物理系的仪器用完了就拆，拆了又装。跑警报的日子，
师生们在郊外的山沟里继续讨论功课，有人带着论文字典，有人带着未完成的实验
记录。那些年在昆明写成的论文，后来散落在世界各地，但其中许多的初稿，是在
铁皮屋顶下、油灯旁边、山沟里的石板上完成的。""",
    """{
  "session": {"id": "a4f7c2e1", "started": "2026-04-03T09:14:22Z", "region": "eu-west-1"},
  "customer": {"tier": "enterprise", "seats": 1240, "renewal": "2027-01-15"},
  "usage": {
    "inference": {"requests": 8421937, "tokens_in": 1120394221, "tokens_out": 402118336},
    "training": {"jobs": 217, "gpu_hours": 18432.5, "checkpoint_gb": 91.3}
  },
  "incidents": [
    {"id": "INC-2201", "severity": 2, "started": "2026-03-28T21:04:11Z",
     "duration_minutes": 43, "affected": ["inference", "dashboard"],
     "cause": "upstream capacity provider failure", "resolved": true},
    {"id": "INC-2207", "severity": 3, "started": "2026-04-01T02:33:57Z",
     "duration_minutes": 11, "affected": ["batch-ingest"], "resolved": true}
  ],
  "capacity": {"gpu_allocated": 512, "gpu_used_peak": 489, "queue_depth_p99": 74}
}""",
    """El camión arrancó al amanecer y tomó la carretera de tierra que subía hacia la
sierra. En la caja, entre sacos de maíz y bidones de agua, iban once personas y
dos gallos. El conductor, que había hecho ese viaje dos veces por semana durante
veinte años, no necesitaba mirar el camino: sabía el nombre de cada curva, de
cada zanja, de cada árbol caído que había que esquivar. A media mañana pararon
en un paraje donde había una cruz de madera y una pileta de agua clara. Allí
comieron tortillas con frijoles y queso, y el más viejo del grupo contó otra vez
la historia del túnel de la mina, la fiebre del oro del año cuarenta y ocho, y
cómo el río cambió de curso una noche de tormenta y se llevó la mitad del pueblo
viejo. Nadie lo interrumpía, aunque todos se la sabían de memoria.""",
    """In compiler design, register allocation is traditionally formulated as a graph
coloring problem: build an interference graph whose nodes are program variables
(live ranges) and whose edges connect variables live at the same program point,
then color the graph with k colors, where k is the number of available registers.
Chaitin's allocator spills a variable when no color can be found, inserting load
and store instructions around the definition and uses, then rebuilding the graph
because spilling changes live ranges. Linear-scan allocation, by contrast, sorts
live intervals by start position and walks them once, keeping an active list and
evicting the interval with the furthest end point when a register is needed -
a single pass with worse register quality but predictable, fast compilation,
which made it the standard choice for just-in-time compilers. SSA form makes
interference sparser: variables defined once interfere only when their live
ranges overlap, and the dominance ordering of definitions enables coalescing of
copy instructions through parallel-copy elimination and lost-copy avoidance.""",
    """# Migration notes: v4 to v5

The v5 release rewrites the storage engine. Read this before upgrading.

**Breaking changes**

1. The manifest format changed from YAML to a length-prefixed binary envelope.
   Run `ninfer migrate --in place/ --out place2/` once; it is idempotent and
   verifies every page checksum before touching the destination.
2. `GET /v4/objects` is removed. Use `GET /v5/objects?cursor=...`; responses
   now return at most 500 names and include a `next_cursor` field.
3. Timestamps are microseconds since epoch (int64), not ISO strings.

**Deprecations**

- `--sort-key` still works but is ignored; v5 always returns canonical order.
- The `x-retention` header moved to the object metadata envelope.

**Operational checklist**

- [ ] Free disk space >= 2.2x the v4 dataset size
- [ ] Backup of the manifest and at least the two most recent snapshots
- [ ] Drain writers (v5 tolerates one stale writer for at most 300 seconds)
- [ ] After migration: `ninfer verify --deep` (about 40 minutes per terabyte)

Rollback: keep the v4 directory until `verify --deep` passes twice in a row.""",
)

# Transformers module that consumes each NInfer input site. Grouped parents share one input:
# attention query/key/gate/value read the q_proj input, GDN query/key/value/z the in_proj_qkv
# input, and MLP gate/up the gate_proj input.
_SITE_MODULES = {
    "attention/input_projection": "self_attn.q_proj",
    "attention/output_projection": "self_attn.o_proj",
    "gdn/input_projection": "linear_attn.in_proj_qkv",
    "gdn/output_projection": "linear_attn.out_proj",
    "mlp/gate_up_projection": "mlp.gate_proj",
    "mlp/down_projection": "mlp.down_proj",
}


def site_name(layer: int, family: str, parent: str) -> str:
    return f"text/layers/{layer}/{family}/{parent}/input_scale_divisor"


def all_sites(layer_types) -> dict[str, str]:
    """Every projection input site of a Qwen3.5 dense Text model -> Transformers module path."""
    sites = {}
    for layer, kind in enumerate(layer_types):
        mixer = "attention" if kind == "full_attention" else "gdn"
        for family, parent in (
            (mixer, "input_projection"),
            (mixer, "output_projection"),
            ("mlp", "gate_up_projection"),
            ("mlp", "down_projection"),
        ):
            module = _SITE_MODULES[f"{family}/{parent}"]
            sites[site_name(layer, family, parent)] = f"layers.{layer}.{module}"
    return sites


def divisor(amax: float) -> float:
    """binary32(2688 / amax); the corpus must produce a positive finite maximum."""
    if not amax > 0.0 or amax == float("inf"):
        raise ValueError(f"site maximum must be positive and finite, got {amax}")
    return struct.unpack("<f", struct.pack("<f", FULL_RANGE / amax))[0]


class SiteMaxima:
    """Forward pre-hooks that keep a device-side running max|input| per site and bucket."""

    def __init__(self, language_model, sites: dict[str, str]):
        import torch

        self._torch = torch
        modules = dict(language_model.named_modules())
        self.sites = dict(sites)
        self.bucket = None
        self.maxima: dict[str, dict[str, object]] = {}
        self._handles = []
        for name, path in self.sites.items():
            if path not in modules:
                raise KeyError(f"{name}: module {path} is absent from the language model")
            self._handles.append(
                modules[path].register_forward_pre_hook(self._hook(name), with_kwargs=True)
            )

    def _hook(self, name):
        def hook(module, args, kwargs):
            if self.bucket is None:
                return
            hidden = args[0] if args else kwargs.get("input")
            if hidden is None:
                raise RuntimeError(f"{name}: hook captured no input tensor")
            value = hidden.detach().abs().amax().float()
            bucket = self.maxima.setdefault(self.bucket, {})
            previous = bucket.get(name)
            bucket[name] = value if previous is None else self._torch.maximum(previous, value)

        return hook

    def results(self) -> dict[str, dict[str, float]]:
        return {
            bucket: {name: float(value.item()) for name, value in values.items()}
            for bucket, values in self.maxima.items()
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_bf16(model_dir: Path, device: str = "cuda"):
    """Whole-checkpoint BF16 load; returns (model, tokenizer, language_model, lm_head)."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto" if device == "cuda" else device
    )
    model.eval()
    return model, tokenizer, model.model.language_model, model.lm_head


def hf_dataset_file(repository: str, revision: str, filename: str) -> Path:
    """One file of a pinned public Hugging Face dataset revision."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repository, filename, revision=revision, repo_type="dataset"))


def token_windows(ids: list[int], total: int, length: int = WINDOW_TOKENS) -> list[list[int]]:
    """The first ``total`` tokens as independent windows of at most ``length`` tokens."""
    if len(ids) < total:
        raise ValueError(f"source provides {len(ids)} tokens, fewer than {total}")
    ids = ids[:total]
    return [ids[i : i + length] for i in range(0, total, length)]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_windows(corpus: str, tokenizer) -> tuple[list[list[int]], dict]:
    """Token windows of a named calibration corpus and the provenance of its sources.

    ``cometkim-v1`` is the fixed ten-document corpus as one window. ``broad-v1`` adds pinned
    public slices that do not overlap the kld-400k-v1 evaluation corpus (other perplexity
    streams, Belebele languages, MATH-500 rows, s1K-1.1 and UltraChat conversations), rendered
    chat conversations included, so the maxima cover the special-token and reasoning activations
    of real requests.
    """
    fixed = tokenizer("\n\n".join(CALIBRATION_DOCUMENTS), add_special_tokens=False)["input_ids"]
    if corpus == "cometkim-v1":
        return [fixed], {"documents": len(CALIBRATION_DOCUMENTS)}
    if corpus != "broad-v1":
        raise ValueError(f"unknown calibration corpus {corpus!r}")
    import pandas

    windows = [fixed]
    provenance: dict = {"documents": len(CALIBRATION_DOCUMENTS)}
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    repository = Path(__file__).resolve().parents[2]
    for relative in BROAD["streams"]:
        data = (repository / relative).read_bytes()
        provenance[relative] = {"sha256": _sha256(data)}
        windows += token_windows(encode(data.decode("utf-8")), BROAD["stream_tokens"])
    source = BROAD["belebele"]
    for language in source["languages"]:
        path = hf_dataset_file(source["repository"], source["revision"], f"data/{language}.jsonl")
        passages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and (passage := json.loads(line)["flores_passage"]) not in passages:
                passages.append(passage)
        provenance[f"{source['repository']}/{language}"] = {"sha256": _sha256(path.read_bytes())}
        windows += token_windows(encode("\n\n".join(passages)), source["tokens"])
    source = BROAD["math"]
    path = hf_dataset_file(source["repository"], source["revision"], source["file"])
    provenance[source["repository"]] = {"sha256": _sha256(path.read_bytes())}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    text = "\n\n".join(
        f"Problem: {row['problem']}\nSolution: {row['solution']}" for row in rows[source["first_row"] :]
    )
    windows += token_windows(encode(text), source["tokens"])
    source = BROAD["s1k"]
    path = hf_dataset_file(source["repository"], source["revision"], source["file"])
    provenance[source["repository"]] = {"sha256": _sha256(path.read_bytes())}
    frame = pandas.read_parquet(path).iloc[source["rows"][0] : source["rows"][1]]
    for row in frame.itertuples():
        messages = [
            {"role": "user", "content": row.question},
            {
                "role": "assistant",
                "content": row.deepseek_attempt,
                "reasoning_content": row.deepseek_thinking_trajectory,
            },
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        windows.append(list(ids)[:WINDOW_TOKENS])
    source = BROAD["ultrachat"]
    path = hf_dataset_file(source["repository"], source["revision"], source["file"])
    provenance[source["repository"]] = {"sha256": _sha256(path.read_bytes())}
    frame = pandas.read_parquet(path).iloc[source["rows"][0] : source["rows"][1]]
    for row in frame.itertuples():
        messages = [{"role": m["role"], "content": m["content"]} for m in row.messages]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, enable_thinking=False
        )
        windows.append(list(ids)[:WINDOW_TOKENS])
    return windows, provenance


def calibration_document(
    maxima: dict[str, float], sites, *, model, corpus: str, tokens: int, provenance: dict
) -> dict:
    missing = sorted(set(sites) - set(maxima))
    if missing:
        raise ValueError(f"calibration measured no input for {len(missing)} sites: {missing[:3]}")
    return {
        "encoder_profile": "NVFP4_MAXABS_DIVISOR_RNE_V1",
        "divisor_formula": "d_x = binary32(2688 / max|site input| over the calibration corpus)",
        "corpus_id": f"nvfp4-calibration-{corpus}",
        "corpus_tokens": tokens,
        "corpus_provenance": provenance,
        "model": model,
        "measured_sites": {
            name: {"amax": maxima[name], "input_scale_divisor": divisor(maxima[name])}
            for name in sorted(sites)
        },
    }


def measure_corpus(windows, language_model, sites: dict[str, str], bucket: str) -> dict[str, float]:
    """Per-site maximum absolute input over every window, each an independent prefill."""
    import torch

    hooks = SiteMaxima(language_model, sites)
    hooks.bucket = bucket
    try:
        with torch.inference_mode():
            for ids in windows:
                language_model(
                    input_ids=torch.tensor([ids], device=language_model.device), use_cache=False
                )
    finally:
        hooks.close()
    return hooks.results()[bucket]


def calibrate(model_dir: Path, out: Path, *, corpus: str, device: str = "cuda") -> dict:
    from .official_recipes import nvfp4full_local_sites

    if out.exists():
        raise FileExistsError(f"calibration output already exists: {out}")
    started = time.perf_counter()
    _, tokenizer, language_model, _ = load_bf16(model_dir, device)
    layer_types = language_model.config.layer_types
    sites = all_sites(layer_types)
    local = nvfp4full_local_sites(layer_types)
    windows, provenance = corpus_windows(corpus, tokenizer)
    maxima = measure_corpus(windows, language_model, sites, corpus)
    tokens = sum(len(window) for window in windows)
    document = calibration_document(
        {name: maxima[name] for name in local},
        local,
        model=MODEL,
        corpus=corpus,
        tokens=tokens,
        provenance=provenance,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (out.parent / (out.stem + ".all_sites.json")).write_text(
        json.dumps(maxima, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"calibrated {len(local)} sites over {len(windows)} windows, {tokens} tokens in "
        f"{time.perf_counter() - started:.1f}s -> {out}",
        flush=True,
    )
    return document


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="official BF16 checkpoint")
    parser.add_argument("--corpus", choices=("cometkim-v1", "broad-v1"), default="broad-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    calibrate(args.model, args.out, corpus=args.corpus, device=args.device)


if __name__ == "__main__":
    main()
