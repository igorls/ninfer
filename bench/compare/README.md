# Paired comparison harness

`ninfer_bench` changed interface in the merge through `a754fcf3`. The candidate takes text input
and honors EOS, while earlier builds take token counts from the committed corpus. The two can no
longer run one workload. This harness sends identical token prompts through the public Engine in
either build, so both arms of a paired comparison run the same requests.

## What one process measures

Each process loads the model and runs a short prime, then for P512 and P2048 one warmup wave and
three measured waves. A wave submits C copies of the same corpus prefix at once and waits for
all of them.

Fixed settings: 65 outputs with stop tokens off, temperature 0, raw output, BF16 KV, context 4096,
prefill chunk 1024, CUDA graphs on, prefix reuse off. Dense uses the optimized proposal head and
Flash the full head.

`wave_seconds` is submit-to-completion wall time and is the primary metric. Each row reports
generated `ids`, the speculative counters and the Engine's own phase timings.

## Build

Do this in a checkout of each arm. It adds two files and one line to that checkout, and changes
nothing in the Engine.

```bash
git fetch https://github.com/igorls/ninfer.git refs/heads/bench/release-compare
git checkout FETCH_HEAD -- bench/compare
```

Append this line to `bench/inference/benchmarks.cmake`:

```cmake
include("${PROJECT_SOURCE_DIR}/bench/compare/release_compare.cmake")
```

Then build `ninfer_release_compare` in that arm's own build directory, with the same options as
`ninfer_bench`. The include selects the Engine API from the checkout and links what `ninfer_bench`
links.

## Run

```text
ninfer_release_compare <dense|flash> <artifact> <corpus> <C> <K> <capacity-bytes> <mode> <output.json>
```

- `<corpus>` is `bench/fixtures/bench_corpus.ids`, which must hold 65,536 IDs.
- `<K>` is 0 for ordinary decoding or the MTP draft count.
- `<mode>` is `capacity` or `measure`.

Query the candidate's Context capacity once per profile, outside measured processes. Pass 0 as
`<capacity-bytes>` and read `device_bytes` from the output:

```bash
ninfer_release_compare dense model.ninfer corpus.ids 4 3 0 capacity capacity.json
```

Then measure both arms with that value:

```bash
ninfer_release_compare dense model.ninfer corpus.ids 4 3 <device_bytes> measure candidate.json
```

Only the candidate has `capacity` mode, and only the candidate uses `<capacity-bytes>`. The
baseline records the value and always uses explicit KV of 4096 × C.

Set `NINFER_COMPARE_DESKTOP_RESERVE_MIB` to replace the default 8 GiB desktop reserve, in both arms.
A 32 GB card needs this.

## Comparing results

Run fresh process pairs in AB/BA/AB order. For each row, compare `ids`, `rounds`, `drafted_tokens`,
`accepted_tokens`, `fallback_steps` and `accepted_per_position`. Report the median paired change in
`wave_seconds` with its full range.

[Known differences](../../docs/performance.md#disposition-of-the-remaining-items-2026-11-06) are
recorded in the performance notes. Report every difference anyway, because the recorded causes come
from another device.

## Limits

- The candidate refuses Host experts, so a Flash run needs resident experts. The `2b62e8e5` arm
  reports no Host expert bytes, so it cannot check this.
- Prompts are one repeated corpus prefix. Rows in a wave share it, so this is a narrow workload.
- Nsight Systems API tracing reversed which Flash arm was slower. Do not use it to judge timing.

## Verification

| Arm | Linux | MSVC |
|---|---|---|
| `99d7f90f` | An earlier revision ran on RTX PRO 6000 Server Edition | Compiled and linked, not run |
| `e1ab9af3` | An earlier revision ran on RTX PRO 6000 Server Edition | Front-end check only |
| `2b62e8e5` | Not built | Front-end check only |

The earlier revision had no reserve option and required `moe` in the baseline's memory summary.
This revision has not run on any device, and no revision has run on an RTX 5090 or on Windows.
