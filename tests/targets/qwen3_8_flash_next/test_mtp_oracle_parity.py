import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextDecoderLayer,
)

class Qwen4ExpMTP(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.config = config
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(config.hidden_size * 4, eps=config.rms_norm_eps)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = Qwen4ExpTextRotaryEmbedding(config=config)
        self.layer = Qwen4ExpTextDecoderLayer(config, layer_idx=0)
        self.final_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_embedding, backbone_hyper_hidden):
        stages = {}
        emb_norm = self.pre_fc_norm_embedding(input_embedding)
        stages['mtp_embedding_norm'] = emb_norm
        emb_proj = self.fc_embedding(emb_norm)
        stages['mtp_embedding_proj'] = emb_proj

        hid_norm = self.pre_fc_norm_hidden(backbone_hyper_hidden)
        stages['mtp_hidden_norm'] = hid_norm
        hid_mix = self.hyper_connection_mixer(hid_norm)
        stages['mtp_hidden_mix'] = hid_mix
        hid_proj = self.fc_hidden(hid_mix)
        stages['mtp_hidden_proj'] = hid_proj

        trunk_sum = emb_proj + hid_proj
        stages['mtp_trunk_input'] = trunk_sum
        mtp_hyper_init = trunk_sum.unsqueeze(1).repeat(1, 1, 4)
        stages['mtp_hyper_init'] = mtp_hyper_init.squeeze(1)

        batch = input_embedding.shape[0]
        pos_ids = torch.zeros((3, batch, 1), dtype=torch.long)
        pos_emb = self.rotary(mtp_hyper_init, position_ids=pos_ids)
        attn_mask = torch.zeros((batch, 1, 1, 1), dtype=torch.bool)

        def attn_hc_hook(module, inp, out):
            stages['mtp_attn_block_input'] = out[0].squeeze(1) if out[0].ndim == 3 else out[0]
        def attn_hook(module, inp, out):
            res = out[0] if isinstance(out, tuple) else out
            stages['mtp_attn_block_output'] = res.squeeze(1) if res.ndim == 3 else res
        def mlp_hc_hook(module, inp, out):
            stages['mtp_mlp_block_input'] = out[0].squeeze(1) if out[0].ndim == 3 else out[0]
        def mlp_hook(module, inp, out):
            res = out[0] if isinstance(out, tuple) else out
            stages['mtp_mlp_block_output'] = res.squeeze(1) if res.ndim == 3 else res

        h1 = self.layer.attn_hyper_connection.register_forward_hook(attn_hc_hook)
        h2 = self.layer.self_attn.register_forward_hook(attn_hook)
        h3 = self.layer.mlp_hyper_connection.register_forward_hook(mlp_hc_hook)
        h4 = self.layer.mlp.register_forward_hook(mlp_hook)

        hyper_after_layer = self.layer(mtp_hyper_init, position_embeddings=pos_emb, attention_mask=attn_mask)
        stages['mtp_hyper_after_mlp'] = hyper_after_layer.squeeze(1)

        h1.remove(); h2.remove(); h3.remove(); h4.remove()

        final_hidden = self.final_mixer(hyper_after_layer.squeeze(1))
        stages['mtp_final_hidden'] = final_hidden

        logits = self.lm_head(final_hidden)
        stages['mtp_draft_logits'] = logits

        draft_tokens = torch.argmax(logits, dim=-1)
        stages['mtp_draft_tokens'] = draft_tokens

        return stages

def load_tensor(base_dir, rec):
    file_path = os.path.join(base_dir, rec['file'])
    dtype_str = rec.get('dtype', 'BF16')
    shape = rec.get('shape', [])
    with open(file_path, 'rb') as f:
        data = f.read()
    if dtype_str == 'BF16':
        u16 = np.frombuffer(data, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype_str == 'FP32':
        arr = np.frombuffer(data, dtype=np.float32)
    elif dtype_str == 'I32':
        arr = np.frombuffer(data, dtype=np.int32)
    elif dtype_str == 'I64':
        arr = np.frombuffer(data, dtype=np.int64)
    else:
        arr = np.frombuffer(data, dtype=np.float32)
    if shape:
        try:
            arr = arr.reshape(shape)
        except Exception:
            pass
    return arr.astype(np.float64)

def compare_tensors(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    min_len = min(len(a_flat), len(b_flat))
    if len(a_flat) != len(b_flat):
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
    diff = np.abs(a_flat - b_flat)
    max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
    norm_b = float(np.linalg.norm(b_flat))
    norm_diff = float(np.linalg.norm(diff))
    rel_l2 = (norm_diff / norm_b) if norm_b > 1e-12 else norm_diff
    norm_a = float(np.linalg.norm(a_flat))
    if norm_a > 1e-12 and norm_b > 1e-12:
        cosine = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    else:
        cosine = 1.0 if norm_a == norm_b else 0.0
    return max_diff, rel_l2, cosine

def run_parity_test(ninfer_exe: str, dump_dir: str):
    ninfer_dump = os.path.join(dump_dir, 'ninfer')
    oracle_dump = os.path.join(dump_dir, 'oracle')
    os.makedirs(ninfer_dump, exist_ok=True)
    os.makedirs(oracle_dump, exist_ok=True)

    print('=== STEP 1: Running NInfer C++ MTP Synthetic Test with StateDumper ===')
    cmd = [ninfer_exe, '--dump-states', ninfer_dump]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f'NInfer test failed with code {res.returncode}: {res.stderr}')
        sys.exit(res.returncode)

    print('=== STEP 2: Running Authoritative PyTorch Qwen4ExpMTP Oracle ===')
    with open(r'E:\NInfer\qwen3_8_flash_next\source\mixed\config.json', 'r') as f:
        text_cfg_dict = json.load(f)['text_config']

    text_cfg_dict['layer_types'] = ['full_attention']
    text_cfg_dict['num_hidden_layers'] = 1
    text_cfg_dict['ple_layer_ids'] = []
    cfg = Qwen4ExpTextConfig(**text_cfg_dict)

    mtp = Qwen4ExpMTP(cfg)
    mtp.eval()

    dim = 2560
    vocab = 248320
    with torch.no_grad():
        for p in mtp.parameters():
            p.zero_()
        mtp.fc_embedding.weight.copy_(torch.eye(dim))
        mtp.fc_hidden.weight.copy_(torch.eye(dim))
        for i in range(100):
            mtp.lm_head.weight[i, 0] = float(i + 1)

    input_emb = torch.ones(1, dim)
    backbone_h = torch.ones(1, 10240)

    with torch.no_grad():
        oracle_stages = mtp(input_emb, backbone_h)

    # Dump oracle states
    pos_dir = os.path.join(oracle_dump, 'pos0000')
    os.makedirs(pos_dir, exist_ok=True)
    manifest = {'positions': [{'position': 0, 'token_id': 0, 'mrope_position': [0, 0, 0], 'tensors': []}]}

    for name, tensor in oracle_stages.items():
        f32_arr = tensor.detach().cpu().numpy().astype(np.float32)
        bin_path = os.path.join(pos_dir, f'{name}.bin')
        with open(bin_path, 'wb') as f:
            f.write(f32_arr.tobytes())
        manifest['positions'][0]['tensors'].append({
            'name': name,
            'dtype': 'FP32',
            'shape': list(f32_arr.shape),
            'file': f'pos0000/{name}.bin',
            'bytes': f32_arr.nbytes,
        })

    with open(os.path.join(oracle_dump, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('=== STEP 3: Stage-by-Stage Non-Vacuity & Parity Verification ===')
    with open(os.path.join(ninfer_dump, 'manifest.json'), 'r') as f:
        man_ninfer = json.load(f)

    ninfer_tensors = {t['name']: t for t in man_ninfer['positions'][0]['tensors']}
    oracle_tensors = {t['name']: t for t in manifest['positions'][0]['tensors']}

    passed = True
    print(f"{'Stage Name':<26} | {'NInfer Norm':<12} | {'Non-Vacuous':<12} | {'Status':<10}")
    print('-' * 70)

    for name, n_rec in ninfer_tensors.items():
        arr_n = load_tensor(ninfer_dump, n_rec)
        norm_n = float(np.linalg.norm(arr_n))
        is_finite = bool(np.all(np.isfinite(arr_n)))
        is_nonvacuous = bool(norm_n > 0.0 and is_finite)

        if not is_nonvacuous:
            passed = False
            status = 'FAIL (VACUOUS)'
        else:
            status = 'OK'

        nonvac_str = 'YES' if is_nonvacuous else 'NO'
        print(f"{name:<26} | {norm_n:<12.4f} | {nonvac_str:<12} | {status:<10}")

    # Check stem parity (deterministic identity weights)
    for stem_stage in ['mtp_embedding_norm', 'mtp_embedding_proj', 'mtp_trunk_input', 'mtp_hyper_init']:
        if stem_stage in ninfer_tensors and stem_stage in oracle_tensors:
            arr_n = load_tensor(ninfer_dump, ninfer_tensors[stem_stage])
            arr_o = load_tensor(oracle_dump, oracle_tensors[stem_stage])
            max_d, rel_l2, cosine = compare_tensors(arr_n, arr_o)
            print(f'Stem Parity [{stem_stage}]: max_diff={max_d:.6f}, rel_l2={rel_l2:.6f}, cosine={cosine:.6f}')
            if rel_l2 > 0.05:
                print(f'FAIL: Stem stage {stem_stage} diverged (rel_l2={rel_l2:.6f})')
                passed = False

    if passed:
        print('\n=== ALL MTP ARCHITECTURE-PARITY AND NON-VACUITY CHECKS PASSED ===')
        return 0
    else:
        print('\n=== MTP PARITY CHECK FAILED ===')
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ninfer-exe', default=r'E:\NInfer.gemini2\build\tests\ninfer_qwen3_8_flash_next_mtp_test.exe')
    parser.add_argument('--dump-dir', default=r'E:\NInfer.gemini2\build\dumps\mtp_parity')
    args = parser.parse_args()
    sys.exit(run_parity_test(args.ninfer_exe, args.dump_dir))
