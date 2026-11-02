"""Official I/O engines, identical original FP32 adapter merge and head."""
import json
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from shunong_accel._legacy.vdm.models.vdm_model import VDM
from shunong_accel._legacy.eval_visual_jev_merge import exact_adapter
from shunong_accel._legacy.visual_jev_compat import install_metadata_shim
from shunong_accel._legacy.visual_jev_paper_compat import install as install_paper_compat
from shunong_accel._legacy.fundm_sweep_contract import SweepHead
from shunong_accel._legacy.loading_device_runtime import exact_models


def load_profile(c, mode):
    profile = {}
    start = time.perf_counter()

    def stage(name, fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        v = fn()
        torch.cuda.synchronize()
        profile[name] = time.perf_counter() - t
        return v

    m = VDM.__new__(VDM)
    torch.nn.Module.__init__(m)
    m.model_path, m.device_str = c['model'], 'cuda'
    m.processor = stage('processor', lambda: AutoProcessor.from_pretrained(c['model']))
    m.tokenizer = m.processor.tokenizer
    m.config = stage('config', lambda: AutoConfig.from_pretrained(c['model']))
    if mode == 'original':
        m.backbone = stage('weight_from_pretrained', lambda:
            Qwen3VLForConditionalGeneration.from_pretrained(
                c['model'], dtype=torch.bfloat16, attn_implementation='sdpa'))
        stage('to_cuda', lambda: m.backbone.to('cuda'))
    else:
        def empty():
            # Preserve real nonpersistent rotary buffers, while parameters are meta.
            with init_empty_weights(include_buffers=False):
                return Qwen3VLForConditionalGeneration._from_config(
                    m.config, dtype=torch.bfloat16, attn_implementation='sdpa')
        m.backbone = stage('meta_model', empty)

        def weights():
            if mode == 'fastsafetensors':
                from fastsafetensors import fastsafe_open
                index = Path(c['model'])/'model.safetensors.index.json'
                paths = sorted({str(index.parent / n) for n in
                                json.loads(index.read_text())['weight_map'].values()})
                with fastsafe_open(paths, device='cuda:0', nogds=True) as f:
                    state = {k: f.get_tensor(k) for k in f.keys()}
                # HF safetensors omits the tied output alias. Reconstruct only
                # this declared tie, keeping strict checking for every key.
                if m.config.tie_word_embeddings and 'lm_head.weight' not in state:
                    state['lm_head.weight'] = state['model.language_model.embed_tokens.weight']
                m.backbone.load_state_dict(state, strict=True, assign=True)
            elif mode == 'tensorizer':
                from tensorizer import TensorDeserializer
                with TensorDeserializer(c['tensorizer_file'], device='cuda:0', num_readers=4) as f:
                    f.load_into_module(m.backbone)
            elif mode == 'sllm_store':
                from shunong_accel._legacy.loader_store_io import load_state
                state = load_state(c, profile)
                # Store0.8 restores many from_blob views into one cudaMalloc;
                # only the offset-zero tensor owns/free()s that allocation.
                # Keep every original tensor alive across module.float(),
                # otherwise replacing that first parameter frees other views.
                m._loader_state_owner = state
                m.backbone.load_state_dict(state, strict=True, assign=True)
            else:
                raise ValueError(mode)
            m.backbone.tie_weights()
            # Both engines leave the exact nonpersistent buffers available.
            for module in m.backbone.modules():
                for name, buf in module.named_buffers(recurse=False):
                    assert not buf.is_meta, name
                    if not buf.is_cuda:
                        module._buffers[name] = buf.to('cuda')
            assert all(p.is_cuda and not p.is_meta for p in m.backbone.parameters())
            assert all(b.is_cuda for b in m.backbone.buffers())
        stage('weight_engine_to_cuda', weights)
    m.backbone.eval()
    m.heads = None
    m.image_token_id = m.config.image_token_id
    m.keep_full_lm_logits = False
    m._option_ids = m._build_option_ids()
    m.processor.image_processor.max_pixels = 200704
    install_metadata_shim(m)
    install_paper_compat(m)
    m.backbone = stage('attach_adapter', lambda: PeftModel.from_pretrained(
        m.backbone, c['adapter'], local_files_only=True, is_trainable=False).eval().requires_grad_(False))
    stage('exact_adapter_check', lambda: exact_adapter(m, load_file(
        str(Path(c['adapter'])/'adapter_model.safetensors'))))
    buffers = {n: b.detach().clone() for n, b in m.vl.named_buffers()}
    stage('whole_model_fp32', lambda: m.float())
    if hasattr(m, '_loader_state_owner'):
        stage('store_release_shared_storage', lambda: delattr(m, '_loader_state_owner'))
    m.backbone = stage('safe_merge', lambda: m.backbone.merge_and_unload(
        safe_merge=True).eval().requires_grad_(False))
    stage('whole_model_bf16', lambda: m.bfloat16())
    for n, b in m.vl.named_buffers():
        b.data = buffers[n].to(b.device)

    def headload():
        h = SweepHead(m.config.text_config.hidden_size, 'linear').cuda()
        h.load_state_dict(torch.load(c['head'], map_location='cpu', weights_only=False)['state_dict'])
        return h.eval().requires_grad_(False)

    head = stage('head', headload)
    assert len(m.vl_model.language_model.layers) == 36
    assert not any('lora_' in n for n, _ in m.named_parameters())
    profile['total_seconds'] = time.perf_counter()-start
    return m, head, profile
