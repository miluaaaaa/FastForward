"""Full 4B execution variants. No resolution/layer/readout approximation."""
import contextlib
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shunong_accel._legacy.gpu_guard import require_allocation, prove
from shunong_accel._legacy.shunong_loader import load_model
from shunong_accel._legacy.vdm.models.prompts import prefix_text, suffix_text
from shunong_accel._legacy.vision_graph_runtime import VisionGraph
from shunong_accel._legacy.packing_layout_runtime import pack_weights
from shunong_accel._legacy.serving_dynamic_norm import fused_norm
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS


def decode(path):
    path = Path(path)
    data = path.read_bytes()
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        import pyspng
        array = pyspng.load(data)
        if array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 3:
            return Image.fromarray(array, 'RGB')
    # Match PIL's original RGB conversion for other formats/channel layouts.
    import io
    with Image.open(io.BytesIO(data)) as image:
        return image.convert('RGB')


class FullFourB:
    """One resident model. Graphs contain computation, never cached answers/features."""
    def __init__(self, root):
        self.root = Path(root)
        require_allocation()
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        self.config = json.loads((self.root/'config.json').read_text())
        self.model, unused_head = load_model(self.config)
        del unused_head
        self.model.eval().requires_grad_(False)
        m = self.model
        assert len(m.vl_model.language_model.layers) == 36
        assert len(m.vl_model.visual.blocks) == 24
        assert m.heads is None
        self.identity = dict(model_path=self.config['model'], adapter=self.config['adapter'],
            parameters=sum(p.numel() for p in m.parameters()), language_layers=36,
            vision_layers=24, readout='original full-vocabulary LM projection then option selection',
            image_processor_size=dict(m.processor.image_processor.size),
            image_processor_min_pixels=m.processor.image_processor.min_pixels,
            image_processor_max_pixels_attribute=m.processor.image_processor.max_pixels,
            processor_note='Preserve actual size fields; max_pixels assignment alone is ineffective here',
            trained=False, quantized=False)
        self.versions = tuple(p._version for p in m.parameters())
        self.lock = threading.Lock()
        self.vision = VisionGraph(m)
        self.vision.mode = 'static'
        self.packed = None
        self.cache = {'full_graph':{}, 'full_packed':{}}
        self.pools = {k:torch.cuda.graph_pool_handle() for k in self.cache}
        self.builds = []
        self.last = {}

    def prepare(self, image, question, candidates):
        m = self.model
        enc = m.processor(text=[prefix_text('')], images=[image], return_tensors='pt')
        suffix = m.tokenizer(suffix_text(question, candidates, 'choice'),
                             add_special_tokens=False)['input_ids']
        ids = torch.cat((enc['input_ids'][0], torch.tensor(suffix, dtype=torch.long)))
        mm = (ids[None] == m.image_token_id).to(torch.int32)
        # Integer positional metadata only. Pixels/model operators still execute on GPU.
        positions, _ = m.vl_model.get_rope_index(ids[None], mm_token_type_ids=mm,
                                                image_grid_thw=enc['image_grid_thw'])
        return dict(ids=ids, pixels=enc['pixel_values'], grid=enc['image_grid_thw'],
                    positions=positions, visual_indices=mm[0].nonzero().flatten())

    def inputs(self, prepared):
        m = self.model
        pixels = prepared['pixels'].to('cuda', m.dtype)
        key = self.vision.key(prepared['grid'])
        if key in self.vision.cache:
            visual, deep = self.vision.features(pixels, prepared['grid'])
        else:
            # Full original singleton SDPA, without GPU cu_seqlens.tolist() per block.
            # Geometry may be reused; image-dependent embeddings are always recomputed.
            image, deep = self.vision.core(pixels, self.vision.geo(prepared['grid']))
            visual = (image,)
            self.vision.last_fallback = True
        visual = torch.cat(visual)
        indices = prepared['visual_indices'].cuda()
        assert len(indices) == len(visual)
        hidden = m.vl_model.get_input_embeddings()(prepared['ids'][None].cuda())
        hidden[0].index_copy_(0, indices, visual.to(hidden.dtype))
        ds = torch.zeros((len(deep), *hidden.shape), device='cuda', dtype=hidden.dtype)
        for j, values in enumerate(deep):
            ds[j, 0].index_copy_(0, indices, values.to(ds.dtype))
        return hidden, ds, prepared['positions'].cuda()

    def attention(self, a, weight, hidden, rope):
        import torch.nn.functional as F
        shape = hidden.shape[:-1]
        head_shape = (*shape, -1, a.head_dim)
        q, k, v = F.linear(hidden, weight).split(
            [a.q_proj.out_features, a.k_proj.out_features, a.v_proj.out_features], dim=-1)
        # Runtime N: unseen sequence lengths must not compile new Triton kernels.
        q = fused_norm(q.contiguous().view(head_shape), a.q_norm).transpose(1, 2)
        k = fused_norm(k.contiguous().view(head_shape), a.k_norm).transpose(1, 2)
        v = v.view(head_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *rope)
        out, _ = ALL_ATTENTION_FUNCTIONS['sdpa'](a, q, k, v, None,
            dropout=0., scaling=a.scaling)
        return a.o_proj(out.reshape(*shape, -1).contiguous())

    def core(self, args, mode):
        h, deep, positions = args
        m = self.model
        lm = m.vl_model.language_model
        rope = lm.rotary_emb(h, positions)
        cache_position = torch.arange(h.shape[1], device=h.device)
        for j, layer in enumerate(lm.layers):
            if mode == 'full_packed':
                qw, gw = self.packed[j]
                attention = self.attention(layer.self_attn, qw,
                    fused_norm(h, layer.input_layernorm), rope)
                h = h + attention
                import torch.nn.functional as F
                gate, up = F.linear(fused_norm(h, layer.post_attention_layernorm), gw).chunk(2, dim=-1)
                h = h + layer.mlp.down_proj(layer.mlp.act_fn(gate)*up)
            else:
                h = layer(h, position_embeddings=rope, attention_mask=None,
                          position_ids=positions[0], cache_position=cache_position,
                          past_key_values=None, use_cache=False)
            if j < len(deep):
                h = h + deep[j]
        last = lm.norm(h[:, -1]).to(m.dtype)
        return m.lm_head(last).index_select(-1, m.option_token_ids).float()

    def build(self, args, mode):
        n = args[0].shape[1]
        if n in self.cache[mode]:
            return
        start = time.perf_counter()
        static = tuple(x.clone() for x in args)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.core(static, mode)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=self.pools[mode]):
            output = self.core(static, mode)
        torch.cuda.synchronize()
        self.cache[mode][n] = (static, graph, output)
        self.builds.append(dict(mode=mode, tokens=n, seconds=time.perf_counter()-start))

    def prebuild(self, prototypes):
        """Only frozen development-cohort common shapes; no per-request capture."""
        self.packed = pack_weights(tuple(self.model.vl_model.language_model.layers))
        for prepared in prototypes['vision']:
            self.vision.build(prepared['pixels'].to('cuda', self.model.dtype), prepared['grid'].cuda())
        self.vision.mode = 'graph'
        for prepared in prototypes['language']:
            args = self.inputs(prepared)
            for mode in self.cache:
                self.build(args, mode)
        torch.cuda.synchronize()

    def execute(self, prepared, mode):
        args = self.inputs(prepared)
        n = len(prepared['ids'])
        entry = self.cache[mode].get(n)
        if entry is None:
            output = self.core(args, mode)
        else:
            static, graph, output = entry
            for dest, value in zip(static, args):
                dest.copy_(value)
            graph.replay()
        self.last = dict(tokens=n, grid=prepared['grid'].tolist(),
            visual_tokens=len(prepared['visual_indices']), language_graph=entry is not None,
            vision_graph=not self.vision.last_fallback)
        return output

    def prove(self, row):
        with Image.open(row['image_path']) as image:
            g = self.model.prepare_group(image.convert('RGB'),
                [dict(instruction=row['instruction'], candidates=row['candidates'], qtype='choice')])
        return prove(self.model, lambda:self.model.run_independent(g), self.root/'results/gpu')

    @torch.no_grad()
    def predict(self, image, question, candidates, item_id='request', mode='full_graph'):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Nonempty question required')
        if not 2 <= len(candidates) <= 8 or any(not isinstance(x, str) or not x.strip() for x in candidates):
            raise ValueError('Two to eight nonempty choices required')
        if mode not in ['native_full', *self.cache]:
            raise ValueError('Unknown execution mode')
        with self.lock:
            if isinstance(image, (str, Path)):
                if mode == 'native_full':
                    with Image.open(image) as opened:
                        rgb = opened.convert('RGB')
                else:
                    rgb = decode(image)
            elif isinstance(image, Image.Image) and image.mode == 'RGB':
                rgb = image
            else:
                raise ValueError('Provide an original image path or original RGB PIL image')
            if mode == 'native_full':
                group = self.model.prepare_group(rgb,
                    [dict(instruction=question, candidates=list(candidates), qtype='choice')])
                output = self.model.run_independent(group)['lm_option_logits']
                self.last = dict(tokens=len(group.prefix_ids)+len(group.suffix_ids[0]),
                    grid=group.image_grid_thw.tolist(),
                    visual_tokens=int(group.image_grid_thw.prod())//4,
                    language_graph=False, vision_graph=False)
            else:
                output = self.execute(self.prepare(rgb, question, candidates), mode)
            logits = output[0, :len(candidates)].float().cpu().tolist()
            assert all(math.isfinite(x) for x in logits)
            pred = logits.index(max(logits))
            exps = [math.exp(x-max(logits)) for x in logits]
            return dict(id=item_id, answer=dict(letter=chr(65+pred), text=candidates[pred]),
                        logits=logits, probabilities=[x/sum(exps) for x in exps])

    def assert_frozen(self):
        assert self.versions == tuple(p._version for p in self.model.parameters())
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        runtime = FullFourB(args.root)
        # API defaults to eager execution unless caller explicitly provisions graphs.
        runtime.packed = pack_weights(tuple(runtime.model.vl_model.language_model.layers))
    for line in sys.stdin:
        if line.strip():
            obj = json.loads(line)
            result = runtime.predict(obj['image_path'], obj['question'], obj['candidates'], obj.get('id', 'request'))
            print(json.dumps(result, ensure_ascii=False), flush=True)
