# Qwen3.8-Flash-Next Oracle & State Diagnostic Harness

This directory provides the authoritative CPU FP32 reference forward pass and state divergence analysis harness for Qwen3.8-Flash-Next.

## 1. Environment Setup

Run the setup script using Python 3.14 to create the isolated virtual environment:

```powershell
.\setup_env.ps1
```

This creates the isolated venv at `E:\NInfer\venv-qwen4exp` with PyTorch (CPU), safetensors, numpy, and transformers.

## 2. Generating State Dumps

### Deliverable A (NInfer C++ Engine Dump)
Generate raw stage tensor dumps for single token or prompt execution using the reference tool:

- **Single Token Execution (e.g. `<|im_start|>` token 248045):**
  ```powershell
  $env:PATH = "P:\third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:PATH
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --execute-token --token-id 248045 `
    --dump-states P:\dumps\ninfer_tok0
  ```

- **Chat Diagnostic Prompt Prefill:**
  ```powershell
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --chat-diagnostic --prompt "Hello" `
    --dump-states P:\dumps\ninfer_chat
  ```

### Deliverable B (Python CPU Oracle Dump)
Generate reference tensors for the same token(s):

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --token-id 248045 `
  --dump-states P:\dumps\oracle_tok0
```

For multi-token sequence:
```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --tokens "248045,151644,872,198" `
  --dump-states P:\dumps\oracle_seq
```

## 3. Comparing States and Finding First Divergence

Run `compare_states.py` to compare stage tensors across positions:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe compare_states.py `
  P:\dumps\ninfer_tok0 `
  P:\dumps\oracle_tok0 `
  --threshold 0.05
```

The script reports $\max |d|$, $\text{rel-L2}$, and $\text{cosine}$ similarity per stage and immediately isolates the exact layer and operator where the numerical divergence starts.
