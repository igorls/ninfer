# NInfer on Google Colab G4 (RTX PRO 6000 Blackwell, `sm_120a`)

Ephemeral Colab G4 launcher, precompiled-binary packager, and private SSH bridge for running `ninfer-serve` on demand without installing Tailscale on the Colab VM or exposing any public port.

## Prerequisites

- Local [`colab` CLI](https://github.com/googlecolab/colab-cli) authenticated (`colab sessions` works)
- `ssh` (OpenSSH) and [Bun](https://bun.sh) on your workstation
- Optional: `gh` CLI authenticated for publishing precompiled Linux `sm_120a` release tarballs

## 1. Fast launch (~90s with precompiled binary)

```bash
# Default: pulls prebuilt ninfer-serve-linux-x86_64-sm120a.tar.gz from GitHub Releases
# (tag: colab-linux-sm120a) + downloads neroued/Qwen3.8-27B-nvfp4-NInfer in parallel
bun scripts/colab/colab-ninfer.ts up --session ninfer-g4

# Reuse an existing shared G4 session with bounded VRAM (e.g. alongside ComfyUI):
bun scripts/colab/colab-ninfer.ts up --session solaris-g4-1080p \
  --context-capacity 16GiB --max-concurrency 8

# Force a fresh source build (e.g. after updating workstation):
bun scripts/colab/colab-ninfer.ts up --session ninfer-g4 --build
```

## 2. Connect privately (never puts Colab on Tailscale)

### Local workstation access (`http://127.0.0.1:18010/v1`)

```bash
bun scripts/colab/colab-ninfer.ts tunnel --session ninfer-g4 --port 18010
```

### Zero-firewall bridge to any remote SSH / Tailscale host

```bash
# Exposes Colab ninfer-serve privately at http://127.0.0.1:18010/v1 on the target host
# via two outbound SSH hops (zero inbound firewall rules, zero Tailscale on Colab):
bun scripts/colab/colab-ninfer.ts bridge --session ninfer-g4 --target root@<host> --port 18010
```

> **Note:** Google Colab allows only one active `colab ssh` connection per runtime at a time. Close `tunnel` / `bridge` before running `package` or `stop-server` against the same session.

## 3. Publish a precompiled Linux `sm_120a` binary for future fast boots

After a `--build` run on a Colab G4 VM, package `ninfer-serve` + its CUDA 13.3 runtime `.so` dependencies + `build-info.json` and upload it to the `colab-linux-sm120a` GitHub Release on `igorls/ninfer`:

```bash
bun scripts/colab/colab-ninfer.ts package \
  --session ninfer-g4 \
  --publish-release colab-linux-sm120a
```

## 4. Clean teardown

```bash
# Stop ONLY ninfer-serve (safe when sharing a Colab runtime with another workload):
bun scripts/colab/colab-ninfer.ts stop-server --session solaris-g4-1080p

# Or terminate a dedicated Colab runtime completely:
bun scripts/colab/colab-ninfer.ts down --session ninfer-g4
```
