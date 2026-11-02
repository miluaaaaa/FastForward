"""Private, explicitly approximate structural variants of the resident scorer.

Original checkpoint tensors are never changed. Every graph consumes fresh RGB,
question tokens and candidate positions. Structural choices are frozen at setup.
"""
import copy
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from shunong_accel._legacy.search_runtime import SearchJoint
from shunong_accel._legacy.search_kernels import swiglu
from shunong_accel._legacy.sub10_runtime import text_parts


def spread(indices, count):
    assert 1 <= count <= len(indices)
    if count == 1:
        return [indices[-1]]
    return [indices[i * (len(indices)-1)//(count-1)] for i in range(count)]


class VisionProxy:
    """Redirect only this runtime's merger, leaving shared model modules intact."""
    def __init__(self,visual,ratio=1.,linear=False):
        self.visual=visual;module=visual.merger;self.module=module
        if linear:
            self.weight=(module.linear_fc2.weight.float()@module.linear_fc1.weight.float()).to(module.linear_fc1.weight.dtype)
            self.bias=(module.linear_fc2.weight.float()@module.linear_fc1.bias.float()+module.linear_fc2.bias.float()).to(self.weight.dtype)
            self.first=None
        else:
            n=max(128,int(module.hidden_size*ratio)//128*128)
            self.first=module.linear_fc1.weight[:n].contiguous()
            self.first_bias=module.linear_fc1.bias[:n].contiguous()
            self.weight=module.linear_fc2.weight[:,:n].contiguous()
            self.bias=module.linear_fc2.bias

    def __getattr__(self,name):
        return getattr(self.visual,name)

    def merger(self,x):
        m=self.module
        x=m.norm(x.view(-1,m.hidden_size) if m.use_postshuffle_norm else x).view(-1,m.hidden_size)
        if self.first is not None:x=m.act_fn(F.linear(x,self.first,self.first_bias))
        return F.linear(x,self.weight,self.bias)


class AggressiveJoint(SearchJoint):
    def configure(self, spec, prototype):
        spec = copy.deepcopy(spec)
        f = self.features
        if 'side' in f:
            spec['side'] = f['side']
        if 'language_count' in f:
            n = f['language_count']
            indices = spec['language_indices']
            spec['language_indices'] = (indices[:n] if f.get('language_mode') == 'prefix'
                                         else spread(indices, n))
        if 'vision_count' in f:
            spec['vision_indices'] = (spread(spec['vision_indices'], f['vision_count'])
                                       if f['vision_count'] else [])
        if 'readout' in f:
            spec['readout'] = f['readout']
        super().configure(spec, prototype)
        self.selective_tail = bool(f.get('selective_tail'))
        self.skip_ffn = set()
        if f.get('skip_ffn') == 'alternate':
            self.skip_ffn = set(self.language_indices[1::2])
        elif f.get('skip_ffn') == 'all':
            self.skip_ffn = set(self.language_indices)
        self.narrow = {}
        ratio = f.get('ffn_ratio', 1.)
        assert 0 < ratio <= 1
        if ratio < 1:
            for j in self.language_indices:
                mlp = self.m.vl_model.language_model.layers[j].mlp
                w = self.packed[j][1]
                gate, up = w.chunk(2, dim=0)
                n = max(128, int(gate.shape[0]*ratio)//128*128)
                self.narrow[id(mlp)] = (torch.cat((gate[:n],up[:n])).contiguous(),
                                         mlp.down_proj.weight[:,:n].contiguous())
        # Optional bounded graphs cover the declared deployment length. There
        # is no online compile, cached image or per-evaluation-row warmup.
        if f.get('graph_limit'):
            self.pos = {b:p for b,p in self.pos.items() if b <= f['graph_limit']}
        assert self.pos
        if f.get('merger_ratio',1.)<1. or f.get('merger_linear'):
            self.vg.v=VisionProxy(self.vg.v,f.get('merger_ratio',1.),f.get('merger_linear',False))
        self.pruned_attention={}
        if f.get('attention_heads'):
            for j in self.language_indices:
                a=self.m.vl_model.language_model.layers[j].self_attn
                d=a.head_dim;nq=a.q_proj.out_features//d;nk=a.k_proj.out_features//d
                count=f['attention_heads'];assert 1<=count<=nq and nq%nk==0
                group=nq//nk
                if count<=nk:
                    assert nk%count==0
                    keys=list(range(0,nk,nk//count));queries=[k*group for k in keys]
                else:
                    assert count%nk==0 and count//nk<=group
                    keys=list(range(nk));queries=[k*group+r for k in keys for r in range(count//nk)]
                qr=torch.tensor([h*d+i for h in queries for i in range(d)],device='cuda')
                kr=torch.tensor([h*d+i for h in keys for i in range(d)],device='cuda')
                weight=torch.cat((a.q_proj.weight.index_select(0,qr),a.k_proj.weight.index_select(0,kr),a.v_proj.weight.index_select(0,kr))).contiguous()
                output=a.o_proj.weight.index_select(1,qr).contiguous()
                self.pruned_attention[id(a)]=(weight,output,len(queries),len(keys))
        self.fp8_attention={};self.fp8_mlp={}
        if f.get('fp8'):
            from shunong_accel._legacy.fp8_private import FP8Linear
            for j in self.language_indices:
                layer=self.m.vl_model.language_model.layers[j]
                qw,gw=self.packed[j]
                if f['fp8'] in ('all','attention'):
                    a=layer.self_attn
                    qw,ow,_,_=self.pruned_attention.get(id(a),(qw,a.o_proj.weight,0,0))
                    self.fp8_attention[id(a)]=(FP8Linear(qw),FP8Linear(ow))
                if f['fp8'] in ('all','mlp') and j not in self.skip_ffn:
                    gate,down=self.narrow.get(id(layer.mlp),(gw,layer.mlp.down_proj.weight))
                    self.fp8_mlp[id(layer.mlp)]=(FP8Linear(gate),FP8Linear(down))
        self.spec['base_profile_id']=self.spec.pop('id',None)
        self.spec['effective_norm']='one-pass approximate RMSNorm' if f.get('single_norm') else 'historical exact RMSNorm'
        self.spec['skipped_ffn']=sorted(self.skip_ffn)
        self.spec['ffn_ratio']=ratio
        self.spec['attention_heads']=f.get('attention_heads','all')
        self.spec['merger_ratio']=f.get('merger_ratio',1.)
        self.spec['merger_linear']=bool(f.get('merger_linear'))
        self.spec['numeric_format']='FP8 '+f['fp8'] if f.get('fp8') else 'BF16'

    def prepare(self,row,backend):
        if not self.features.get('rust_tokenizer'):
            return super().prepare(row,backend)
        import shunong_accel._legacy.sub10_runtime as sub10_runtime
        start=time.perf_counter_ns()
        rgb=sub10_runtime.decode(row['image_path'],self.side,backend)
        decoded=time.perf_counter_ns()
        prefix,suffix=text_parts(row,self.style)
        encoding=self.m.tokenizer.backend_tokenizer.encode(suffix,add_special_tokens=False)
        ids=np.concatenate((self.prefix_array,np.asarray(encoding.ids,dtype=np.int64)))
        picks=np.zeros(9,dtype=np.int64)
        cursor=suffix.index('\nOptions:\n')+len('\nOptions:\n')
        for i,opt in enumerate(row['candidates']):
            line=f'{chr(65+i)}. {opt}'
            assert suffix[cursor:cursor+len(line)]==line
            token=encoding.char_to_token(cursor+len(line.rstrip())-1)
            assert token is not None
            picks[i]=len(self.prefix_array)+token
            cursor+=len(line)+1
        picks[8]=len(ids)-1
        return dict(rgb=rgb,ids=torch.from_numpy(ids),picks=torch.from_numpy(picks),k=len(row['candidates']),
            prompt_text=prefix+suffix,decoded_ns=decoded,prepare_ns=time.perf_counter_ns(),start_ns=start)

    def mlp(self, module, w, x):
        if id(module) in self.fp8_mlp:
            gateup,down=self.fp8_mlp[id(module)]
            gate,up=gateup(x).chunk(2,dim=-1)
            product=swiglu(gate,up,self.features.get('mlp_block',512),self.features.get('mlp_warps',4))
            return down(product)
        if id(module) not in self.narrow:
            return super().mlp(module,w,x)
        packed, down = self.narrow[id(module)]
        gate, up = F.linear(x,packed).chunk(2,dim=-1)
        product = swiglu(gate,up,self.features.get('mlp_block',512),
                         self.features.get('mlp_warps',4))
        return F.linear(product,down)

    def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
        if id(a) not in self.fp8_attention and id(a) not in self.pruned_attention:
            return super().attention(a,w,hidden_states,position_embeddings,attention_mask,**kwargs)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS
        from shunong_accel._legacy.search_kernels import norm_rope_pair
        if id(a) in self.fp8_attention:qkv,output=self.fp8_attention[id(a)]
        else:qkv,output,_,_=self.pruned_attention[id(a)]
        shape=hidden_states.shape[:-1];head_shape=(*shape,-1,a.head_dim)
        if id(a) in self.pruned_attention:
            _,_,nq,nk=self.pruned_attention[id(a)]
            splits=[nq*a.head_dim,nk*a.head_dim,nk*a.head_dim]
            attention_module=SimpleNamespace(num_key_value_groups=nq//nk,is_causal=True)
        else:
            splits=[a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features];attention_module=a
        projected=F.linear(hidden_states,qkv) if isinstance(qkv,torch.Tensor) else qkv(hidden_states)
        q,k,v=projected.split(splits,dim=-1)
        q,k=norm_rope_pair(q.view(head_shape),k.view(head_shape),a.q_norm,a.k_norm,*position_embeddings,
            joint=True,block=self.features.get('qk_block',512))
        v=v.view(head_shape).transpose(1,2)
        out,_=ALL_ATTENTION_FUNCTIONS['sdpa'](attention_module,q,k,v,attention_mask,dropout=0.,scaling=a.scaling)
        flattened=out.reshape(*shape,-1).contiguous()
        return (F.linear(flattened,output) if isinstance(output,torch.Tensor) else output(flattened)),None

    def core_joint(self, args, native=False):
        if native:
            return super().core_joint(args,True)
        rgb, ids, picks, mask = args
        lm = self.m.vl_model.language_model
        visual, deep = self.vision(rgb)
        h = self.m.vl_model.get_input_embeddings()(ids)
        h[0].index_copy_(0,self.vi,visual)
        pos = self.pos[ids.shape[1]]
        rope = lm.rotary_emb(h,pos)
        cache = torch.arange(ids.shape[1],device='cuda')
        normalized = self.norm(h,lm.layers[self.language_indices[0]].input_layernorm)
        chosen = None
        for offset,j in enumerate(self.language_indices):
            layer = lm.layers[j]
            qw, gw = self.packed[j]
            final = offset == len(self.language_indices)-1
            injection = deep[j] if j < len(deep) else None
            if final and self.features.get('terminal_queries') and injection is None:
                chosen=self.terminal(layer,qw,gw,h,normalized,rope,picks)
                break
            attn,_ = self.attention(layer.self_attn,qw,hidden_states=normalized,
                position_embeddings=rope,attention_mask=None,position_ids=pos[0],
                cache_position=cache,use_cache=False)
            if j in self.skip_ffn:
                h = self.inject_reference(h+attn,injection)
                if not final:
                    normalized = self.norm(h,lm.layers[self.language_indices[offset+1]].input_layernorm)
                continue
            residual, post = self.post_attention(h,attn,layer)
            if final and self.selective_tail and injection is None:
                selected = picks if self.readout == 'head' else picks[8:9]
                chosen = residual[0].index_select(0,selected) + self.mlp(
                    layer.mlp,gw,post[0].index_select(0,selected))
            else:
                mlp = self.mlp(layer.mlp,gw,post)
                if final:
                    h = self.inject_reference(residual+mlp,injection)
                else:
                    h,normalized = self.carry(residual,mlp,injection,
                        lm.layers[self.language_indices[offset+1]].input_layernorm,self.image_start,self.Nv)
        if chosen is None:
            chosen = h[0].index_select(0,picks)
            last = chosen[8][None]
        else:
            last = chosen[8][None] if self.readout == 'head' else chosen
        if self.readout == 'head':
            return self.heads[str(self.depth)].score(chosen[8][None],chosen[:8][None],mask)
        return F.linear(lm.norm(last).to(self.m.dtype),self.option_weight).float()
