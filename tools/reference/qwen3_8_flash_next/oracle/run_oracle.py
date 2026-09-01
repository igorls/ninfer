import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextModel,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextDecoderLayer,
)

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

class LazyPLEEmbedding(nn.Module):
    def __init__(self, ple_dir: str):
        super().__init__()
        self.ple_dir = ple_dir
        self.shards = {}
        self.weight = torch.empty(0, device="cpu")

    def get_shard(self, shard_idx: int):
        if shard_idx not in self.shards:
            path = os.path.join(self.ple_dir, f"shard_{shard_idx}.safetensors")
            self.shards[shard_idx] = safetensors.safe_open(path, framework="pt", device="cpu")
        return self.shards[shard_idx]

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        # ngram_ids: [batch, seq_len, 16]
        orig_shape = ngram_ids.shape
        flat_ids = ngram_ids.reshape(-1)
        out_list = []
        for r_tensor in flat_ids:
            r = int(r_tensor.item())
            shard_idx = r // 2500012
            local_row = r % 2500012
            shard = self.get_shard(shard_idx)
            i4 = shard.get_tensor("weight_i4")[local_row].to(torch.int32)  # [80] bytes
            scale = shard.get_tensor("weight_scale")[local_row].float()  # [10]
            low = (i4 & 0x0F).float() - 8.0
            high = (i4 >> 4).float() - 8.0
            unpacked = torch.stack([low, high], dim=-1).reshape(160)
            scale_exp = scale.repeat_interleave(16)
            out_list.append(unpacked * scale_exp)
        res = torch.stack(out_list, dim=0).reshape(*orig_shape, 160)
        return res

class LazyExperts(nn.Module):
    def __init__(self, config, layer_idx: int, model_dir: str):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx
        self.model_dir = model_dir
        self.exp_f = None
        self.cache_gate_up = {}
        self.cache_down = {}

    def get_exp_file(self):
        if self.exp_f is None:
            path = os.path.join(self.model_dir, f"ct-experts-layer{self.layer_idx:02d}.safetensors")
            self.exp_f = safetensors.safe_open(path, framework="pt", device="cpu")
        return self.exp_f

    def dequant_matrix(self, prefix: str) -> torch.Tensor:
        f = self.get_exp_file()
        packed = f.get_tensor(prefix + ".weight_packed")
        scale = f.get_tensor(prefix + ".weight_scale").float()
        glob = f.get_tensor(prefix + ".weight_global_scale").item()
        rows, half_cols = packed.shape
        cols = half_cols * 2
        low = (packed & 0x0F).to(torch.long)
        high = (packed >> 4).to(torch.long)
        val_low = FP4_LUT[low]
        val_high = FP4_LUT[high]
        unpacked = torch.stack([val_low, val_high], dim=-1).reshape(rows, cols)
        full_scale = scale.repeat_interleave(16, dim=-1) / glob
        return unpacked * full_scale

    def get_expert_weights(self, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if expert_idx not in self.cache_gate_up:
            w_gate = self.dequant_matrix(f"model.language_model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.gate_proj")
            w_up = self.dequant_matrix(f"model.language_model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.up_proj")
            w_down = self.dequant_matrix(f"model.language_model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.down_proj")
            gate_up = torch.cat([w_gate, w_up], dim=0)  # gate rows FIRST
            self.cache_gate_up[expert_idx] = gate_up
            self.cache_down[expert_idx] = w_down
        return self.cache_gate_up[expert_idx], self.cache_down[expert_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up, down = self.get_expert_weights(expert_idx)
            gate, up = F.linear(current_state, gate_up).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        # Each expert is used at most once per forward; a persistent FP32 cache (~20 MB per
        # expert) exhausts host memory on multi-token prompts.
        self.cache_gate_up.clear()
        self.cache_down.clear()
        return final_hidden_states

def build_oracle(model_dir: str, ple_dir: str):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        root_cfg = json.load(f)
    text_cfg_dict = root_cfg["text_config"]
    cfg = Qwen4ExpTextConfig(**text_cfg_dict)

    with torch.device("meta"):
        model = Qwen4ExpTextModel(cfg)

    # Replace heavy layers before allocating memory on CPU
    for l in range(48):
        model.layers[l].mlp.experts = LazyExperts(cfg, l, model_dir)
    model.layers[1].ple.ple_embedding.ngram_embedding = LazyPLEEmbedding(ple_dir)

    model.to_empty(device="cpu")
    model.float()
    model.eval()

    # Load weights
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    open_safetensors = {}
    def get_raw_tensor(name: str):
        fn = weight_map[name]
        if fn not in open_safetensors:
            p = os.path.join(model_dir, fn)
            open_safetensors[fn] = safetensors.safe_open(p, framework="pt", device="cpu")
        return open_safetensors[fn].get_tensor(name)

    sd = {}
    lm_head_weight = None

    fp8_targets = [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]

    for orig_key, fn in weight_map.items():
        if fn.startswith("ple-bf16-") or "ngram_embedding.weight" in orig_key:
            continue
        if orig_key == "lm_head.weight":
            lm_head_weight = get_raw_tensor(orig_key).float()
            continue
        if not orig_key.startswith("model.language_model."):
            continue
        
        target_key = orig_key[len("model.language_model."):]

        # Skip expert weights
        if ".mlp.experts." in target_key:
            continue

        # Check if FP8 row scale
        is_fp8 = False
        for tgt in fp8_targets:
            if target_key.endswith(tgt + ".weight"):
                is_fp8 = True
                base_name = orig_key[:-len(".weight")]
                w_packed = get_raw_tensor(base_name + ".weight")
                w_scale = get_raw_tensor(base_name + ".weight_scale").float()
                w_f32 = w_packed.to(torch.float32)
                if w_scale.ndim == 1:
                    w_f32 = w_f32 * w_scale.unsqueeze(1)
                else:
                    w_f32 = w_f32 * w_scale
                sd[target_key] = w_f32
                break
            elif target_key.endswith(tgt + ".weight_scale"):
                is_fp8 = True  # handled with .weight
                break

        if not is_fp8:
            t = get_raw_tensor(orig_key)
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            sd[target_key] = t

    res = model.load_state_dict(sd, strict=False)
    
    missing_set = set(res.missing_keys)
    unexpected_set = set(res.unexpected_keys)

    assert len(missing_set) == 0, f"Missing keys found: {missing_set}"
    assert len(unexpected_set) == 0, f"Unexpected keys found: {unexpected_set}"

    # Assert PLE index buffers
    ple_emb = model.layers[1].ple.ple_embedding
    expected_multipliers = [23703573157769, 20109073645365, 8052911324071]
    expected_offsets = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
    expected_vocab_sizes = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

    assert ple_emb.layer_multipliers.tolist() == expected_multipliers, "PLE layer_multipliers mismatch!"
    assert ple_emb.ngram_heads_offsets.tolist() == expected_offsets, "PLE ngram_heads_offsets mismatch!"
    assert ple_emb.ngram_heads_vocab_sizes.tolist() == expected_vocab_sizes, "PLE ngram_heads_vocab_sizes mismatch!"

    return model, lm_head_weight

def register_hooks(model: nn.Module):
    stage_outputs = {}

    def save_output(name: str):
        def hook(module, input, output):
            val = output[0] if isinstance(output, (tuple, list)) else output
            stage_outputs[name] = val.detach().clone()
        return hook

    def save_input(name: str):
        def hook(module, input):
            val = input[0] if isinstance(input, (tuple, list)) else input
            stage_outputs[name] = val.detach().clone()
        return hook

    # Embedding
    model.embed_tokens.register_forward_hook(save_output("embedding"))

    # Layer 1 PLE injection
    model.layers[1].ple.register_forward_hook(save_output("ple_injection"))

    for l in range(48):
        prefix = f"L{l:02d}_"
        # attn_hyper_connection input & output
        model.layers[l].attn_hyper_connection.register_forward_pre_hook(save_input(prefix + "hyper_in"))
        model.layers[l].attn_hyper_connection.register_forward_hook(save_output(prefix + "attn_block_input"))

        # linear_attn or self_attn
        if hasattr(model.layers[l], "linear_attn"):
            model.layers[l].linear_attn.register_forward_hook(save_output(prefix + "attn_block_output"))
        elif hasattr(model.layers[l], "self_attn"):
            model.layers[l].self_attn.register_forward_hook(save_output(prefix + "attn_block_output"))

        # mlp_hyper_connection input & output
        model.layers[l].mlp_hyper_connection.register_forward_pre_hook(save_input(prefix + "hyper_after_attn"))
        model.layers[l].mlp_hyper_connection.register_forward_hook(save_output(prefix + "mlp_block_input"))

        # mlp
        model.layers[l].mlp.register_forward_hook(save_output(prefix + "mlp_block_output"))

        # layer output
        model.layers[l].register_forward_hook(save_output(prefix + "hyper_after_mlp"))

    # Final mixer
    model.hyper_connection_mixer.register_forward_hook(save_output("final_hidden"))

    return stage_outputs

class Qwen4ExpMTP(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.config = config
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = Qwen4ExpTextRotaryEmbedding(config=config)
        self.layer = Qwen4ExpTextDecoderLayer(config, layer_idx=0)
        self.final_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_embedding: torch.Tensor, backbone_hyper_hidden: torch.Tensor):
        stages = {}
        emb_norm = self.pre_fc_norm_embedding(input_embedding)
        stages["mtp_embedding_norm"] = emb_norm
        emb_proj = self.fc_embedding(emb_norm)
        stages["mtp_embedding_proj"] = emb_proj

        hid_mix = self.hyper_connection_mixer(backbone_hyper_hidden)
        stages["mtp_hidden_mix"] = hid_mix
        hid_proj = self.fc_hidden(hid_mix)
        stages["mtp_hidden_proj"] = hid_proj

        trunk_sum = emb_proj + hid_proj
        stages["mtp_trunk_input"] = trunk_sum
        mtp_hyper_init = trunk_sum.unsqueeze(1).repeat(1, 1, 4)
        stages["mtp_hyper_init"] = mtp_hyper_init.squeeze(1)

        batch = input_embedding.shape[0]
        pos_ids = torch.zeros((3, batch, 1), dtype=torch.long)
        pos_emb = self.rotary(mtp_hyper_init, position_ids=pos_ids)
        attn_mask = torch.zeros((batch, 1, 1, 1), dtype=torch.bool)

        def attn_hc_hook(module, inp, out):
            stages["mtp_attn_block_input"] = out[0].squeeze(1) if out[0].ndim == 3 else out[0]
        def attn_hook(module, inp, out):
            res = out[0] if isinstance(out, tuple) else out
            stages["mtp_attn_block_output"] = res.squeeze(1) if res.ndim == 3 else res
        def mlp_hc_hook(module, inp, out):
            stages["mtp_mlp_block_input"] = out[0].squeeze(1) if out[0].ndim == 3 else out[0]
        def mlp_hook(module, inp, out):
            res = out[0] if isinstance(out, tuple) else out
            stages["mtp_mlp_block_output"] = res.squeeze(1) if res.ndim == 3 else res

        h1 = self.layer.attn_hyper_connection.register_forward_hook(attn_hc_hook)
        h2 = self.layer.self_attn.register_forward_hook(attn_hook)
        h3 = self.layer.mlp_hyper_connection.register_forward_hook(mlp_hc_hook)
        h4 = self.layer.mlp.register_forward_hook(mlp_hook)

        hyper_after_layer = self.layer(mtp_hyper_init, position_embeddings=pos_emb, attention_mask=attn_mask)
        stages["mtp_hyper_after_mlp"] = hyper_after_layer.squeeze(1)

        h1.remove(); h2.remove(); h3.remove(); h4.remove()

        final_hidden = self.final_mixer(hyper_after_layer.squeeze(1))
        stages["mtp_final_hidden"] = final_hidden

        logits = self.lm_head(final_hidden)
        stages["mtp_draft_logits"] = logits

        draft_tokens = torch.argmax(logits, dim=-1)
        stages["mtp_draft_tokens"] = draft_tokens

        return stages

def main():
    parser = argparse.ArgumentParser(description="Authoritative HF Qwen4Exp CPU FP32 reference oracle for Qwen3.8-Flash-Next")
    parser.add_argument("--model-dir", default=r"E:\NInfer\qwen3_8_flash_next\source\mixed", help="Path to mixed source model dir")
    parser.add_argument("--ple-dir", default=r"E:\NInfer\qwen3_8_flash_next\source\ple\ples_int4", help="Path to PLE INT4 shards")
    parser.add_argument("--token-id", type=int, default=248045, help="Single token ID to execute")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated token IDs to execute in sequence")
    parser.add_argument("--dump-states", type=str, default="", help="Directory to dump state tensors and manifest")
    parser.add_argument("--mtp-synthetic", action="store_true", help="Run MTP synthetic architecture parity step")
    args = parser.parse_args()

    if args.mtp_synthetic:
        print("Running authoritative Qwen4ExpMTP synthetic reference step ...")
        with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
            text_cfg_dict = json.load(f)["text_config"]
        text_cfg_dict["layer_types"] = ["full_attention"]
        text_cfg_dict["num_hidden_layers"] = 1
        text_cfg_dict["ple_layer_ids"] = []
        cfg = Qwen4ExpTextConfig(**text_cfg_dict)

        mtp = Qwen4ExpMTP(cfg)
        mtp.eval()

        dim = 2560
        with torch.no_grad():
            mtp.fc_embedding.weight.copy_(torch.eye(dim))
            mtp.fc_hidden.weight.copy_(torch.eye(dim))
            mtp.lm_head.weight.zero_()
            for i in range(100):
                mtp.lm_head.weight[i, 0] = float(i + 1)

        input_emb = torch.ones(1, dim)
        backbone_h = torch.ones(1, 10240)

        with torch.no_grad():
            stages = mtp(input_emb, backbone_h)

        print(f"MTP Reference Output: Draft Token = {stages['mtp_draft_tokens'][0].item()}")
        if args.dump_states:
            pos_dir = os.path.join(args.dump_states, "pos0000")
            os.makedirs(pos_dir, exist_ok=True)
            manifest = {"positions": [{"position": 0, "token_id": 0, "mrope_position": [0, 0, 0], "tensors": []}]}
            for name, tensor in stages.items():
                f32_arr = tensor.detach().cpu().numpy().astype(np.float32)
                bin_path = os.path.join(pos_dir, f"{name}.bin")
                with open(bin_path, "wb") as f:
                    f.write(f32_arr.tobytes())
                manifest["positions"][0]["tensors"].append({
                    "name": name,
                    "dtype": "FP32",
                    "shape": list(f32_arr.shape),
                    "file": f"pos0000/{name}.bin",
                    "bytes": f32_arr.nbytes,
                })
            with open(os.path.join(args.dump_states, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"Dumped MTP oracle states to {args.dump_states}/manifest.json")
        return

    print(f"Building authoritative Transformers Qwen4ExpTextModel from {args.model_dir} ...")
    model, lm_head_weight = build_oracle(args.model_dir, args.ple_dir)
    stage_outputs = register_hooks(model)

    if args.ids:
        token_list = [int(t.strip()) for t in args.ids.split(",") if t.strip()]
    else:
        token_list = [args.token_id]

    input_ids = torch.tensor([token_list], dtype=torch.long)
    print(f"Running teacher-forced forward pass for {len(token_list)} tokens: {token_list} ...")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
        logits_all = F.linear(out.last_hidden_state, lm_head_weight)  # [1, seq_len, vocab_size]

    manifest = {"positions": []}

    for pos in range(len(token_list)):
        tok = token_list[pos]
        logits_pos = logits_all[0, pos]
        top5_vals, top5_ids = torch.topk(logits_pos, 5)

        print(f"\n[Position {pos:04d}] Token {tok}:")
        print(f"  Argmax: {top5_ids[0].item()} (logit: {top5_vals[0].item():.2f})")
        print("  Top-5: ", ", ".join(f"{tid.item()}:{val.item():.2f}" for val, tid in zip(top5_vals, top5_ids)))

        if args.dump_states:
            pos_dir = os.path.join(args.dump_states, f"pos{pos:04d}")
            os.makedirs(pos_dir, exist_ok=True)
            pos_records = []

            # Format stage names
            stages = []
            
            # embedding & hyper_init
            emb_pos = stage_outputs["embedding"][0, pos]
            stages.append(("embedding", emb_pos))
            hyper_init_pos = torch.cat([emb_pos, emb_pos, emb_pos, emb_pos], dim=-1)
            stages.append(("hyper_init", hyper_init_pos))

            if "ple_injection" in stage_outputs:
                ple_pos = stage_outputs["ple_injection"][0, pos]
                stages.append(("ple_injection", ple_pos))

            for l in range(48):
                prefix = f"L{l:02d}_"
                if l == 1:
                    # hyper_after_ple is the input to layer 1 attn_hyper_connection
                    stages.append(("hyper_after_ple", stage_outputs[prefix + "hyper_in"][0, pos]))

                stages.append((prefix + "attn_block_input", stage_outputs[prefix + "attn_block_input"][0, pos]))
                stages.append((prefix + "attn_block_output", stage_outputs[prefix + "attn_block_output"][0, pos]))
                stages.append((prefix + "hyper_after_attn", stage_outputs[prefix + "hyper_after_attn"][0, pos]))
                stages.append((prefix + "mlp_block_input", stage_outputs[prefix + "mlp_block_input"][0, pos]))
                stages.append((prefix + "mlp_block_output", stage_outputs[prefix + "mlp_block_output"][0, pos]))
                stages.append((prefix + "hyper_after_mlp", stage_outputs[prefix + "hyper_after_mlp"][0, pos]))

            stages.append(("final_hidden", stage_outputs["final_hidden"][0, pos]))
            stages.append(("logits", logits_pos))

            for name, tensor in stages:
                f32_bytes = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
                bin_file = f"{name}.bin"
                with open(os.path.join(pos_dir, bin_file), "wb") as f:
                    f.write(f32_bytes)
                pos_records.append({
                    "name": name,
                    "dtype": "FP32",
                    "shape": list(tensor.shape),
                    "file": f"pos{pos:04d}/{bin_file}",
                    "bytes": len(f32_bytes),
                })

            manifest["positions"].append({
                "position": pos,
                "token_id": tok,
                "tensors": pos_records,
            })

    if args.dump_states:
        with open(os.path.join(args.dump_states, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nDumped {len(token_list)} positions to {args.dump_states}/manifest.json")

if __name__ == "__main__":
    main()
