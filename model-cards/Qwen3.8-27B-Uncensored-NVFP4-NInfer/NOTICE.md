# Attribution and conversion notice

This distribution contains a converted model artifact with components from:

1. **Qwen**: Qwen3.8-27B, the underlying model. Source: https://huggingface.co/Qwen/Qwen3.8-27B. License: Apache 2.0.
2. **OrcaRouter**: Qwen3.8-27B-Uncensored-NVFP4, revision `69d21348b2d6c11439fb69368f40414c2256e44e`. Source: https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4. License: Apache 2.0. This supplies the target, Vision, MTP and frontend components.
3. **Inco AI / DFlash authors**: Qwen3.8-27B-DFlash2, revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Source: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2. Project: https://github.com/z-lab/dflash. The source model card declares Apache 2.0. This supplies the speculative companion.

The included LICENSE is copied from the pinned OrcaRouter source without modification. Neither pinned component repository supplies a separate NOTICE file.

The conversion and runtime use the igorls/NInfer fork of Neroued/NInfer:
https://github.com/igorls/ninfer and https://github.com/Neroued/ninfer.

## Changes in this distribution

The source checkpoints are repacked into one NInfer container with the registered identity `qwen3.8-27b-orcarouter/nvfp4`. The source's 168 NVFP4 and 232 FP8 body matrices retain their mixed precision allocation. Token embeddings and the full output head retain the source BF16 bytes. NInfer's target recipe packs MTP, Vision and the optimized Q4 proposal head; the separately sourced DFlash2 companion is converted into the supported W8/BF16 execution layouts. All six source frontend resources are embedded verbatim. No additional model training or abliteration was performed for this conversion.

The companion conversion report replaces local filesystem paths with portable placeholders. Its recorded converter revision is the checkout base at conversion time, when the implementation was still uncommitted. The corresponding converter and runtime implementation is now published in NInfer commit `91ce2f2c73c27ffbff3bd7c43a319f295e9597cc`; the artifact manifest records both facts. The checkpoint bytes have not been changed for publication.

This notice attributes the source components and describes the conversion. It does not imply endorsement or add restrictions to the Apache 2.0 license.
