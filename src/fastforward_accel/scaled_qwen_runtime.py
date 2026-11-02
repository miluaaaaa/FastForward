"""FastForward: shape-derived Qwen3-VL fast path and phase-separated residency."""
import copy
import gc
import io
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
import triton
from transformers import AutoConfig,AutoProcessor,Qwen3VLForConditionalGeneration
from fastforward_accel._legacy.vdm.models.vdm_model import VDM
from fastforward_accel._legacy.visual_jev_compat import install_metadata_shim
from fastforward_accel._legacy.visual_jev_paper_compat import install as install_paper_compat
from fastforward_accel._legacy.sub10_runtime import pack_rgb,fast_norm,text_parts
from fastforward_accel._legacy.broad_runtime import BroadVision
from fastforward_accel._legacy.broad_kernels import qk_pair
from fastforward_accel._legacy.lower_kernels import add_rms
from fastforward_accel._legacy.limit_kernels import StaticFP8Linear,attention_direct,lm_readout
from fastforward_accel._legacy.search_kernels import swiglu
from fastforward_accel._legacy.search_kernels import vision_rope
from fastforward_accel._legacy.vision_graph_runtime import VisionGraph
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS
from fastforward_accel._legacy.recovery_adapter import Repair,FP8STE


class Linear:
    @staticmethod
    def apply(x,op):
        if torch.is_grad_enabled() and x.requires_grad:return FP8STE.apply(x,op)
        return op(x).reshape(*x.shape[:-1],op.weight.shape[0])


class NativeNormVision(VisionGraph):
    """Keep native LayerNorm/adds; fuse only FP32 visual rotary arithmetic."""
    def core(self,pixels,geometry):
        v=self.v;pos,cos,sin=geometry;h=v.patch_embed(pixels)+pos
        n=h.shape[0];h=h.reshape(n,-1);deep=[]
        for j,block in enumerate(v.blocks):
            a=block.attn;x=block.norm1(h)
            q,k,value=a.qkv(x).reshape(n,3,a.num_heads,-1).permute(1,0,2,3).unbind(0)
            q,k=vision_rope(q,k,cos,sin)
            q=q.transpose(0,1).unsqueeze(0);k=k.transpose(0,1).unsqueeze(0);value=value.transpose(0,1).unsqueeze(0)
            z=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,value,attention_mask=None,scaling=a.scaling,dropout=0.,is_causal=False)[0]
            h=h+a.proj(z.reshape(n,-1).contiguous());h=h+block.mlp(block.norm2(h))
            if j in v.deepstack_visual_indexes:deep.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(j)](h))
        return v.merger(h),deep


class Runtime:
    def __init__(self,c):
        self.c=c;self.model=VDM.__new__(VDM);m=self.model;nn.Module.__init__(m)
        m.model_path=c['model'];m.device_str='cuda';m.processor=AutoProcessor.from_pretrained(c['model'],local_files_only=True)
        m.tokenizer=m.processor.tokenizer;m.config=AutoConfig.from_pretrained(c['model'],local_files_only=True)
        m.backbone=Qwen3VLForConditionalGeneration.from_pretrained(c['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval().requires_grad_(False)
        assert all(p.is_cuda for p in m.backbone.parameters()),'No CPU offload allowed'
        m.heads=None;m.image_token_id=m.config.image_token_id;m.keep_full_lm_logits=False;m._option_ids=m._build_option_ids()
        m.processor.image_processor.max_pixels=200704
        install_metadata_shim(m);install_paper_compat(m)
        self.lm=m.vl_model.language_model;self.depth=len(self.lm.layers);self.final=self.depth-1;self.width=m.config.text_config.hidden_size
        assert (self.depth,self.width,len(m.vl_model.visual.blocks))==tuple(c["expected_dimensions"])
        self.tokenizer=m.tokenizer;self.head=m.lm_head.weight.index_select(0,m.option_token_ids[:8]).detach().contiguous()
        self.option_ids=[self.tokenizer.encode(' '+a,add_special_tokens=False)[0] for a in 'ABCDEFGH']
        assert self.option_ids==m.option_token_ids[:8].tolist()
        self.features=dict(vision_rope=True,vision_no_cat=True,vision_add_norm=True,patch_mode=c.get('vision_patch_mode','conv'))
        self.vg=BroadVision(m);self.vg.owner=self;self.vg.active_indices=set(range(len(self.vg.v.blocks)))
        self.geometry={};self.quant=None;self.repair=Repair(self.width,rank=32,seed=731).requires_grad_(False)
        self.cache={};self.side=224;self.indices=list(range(self.depth));self.versions=tuple(p._version for p in m.parameters())
        self.buffers={};self.max_tokens=c['buckets'][-1]

    def frozen(self):
        assert tuple(p._version for p in self.model.parameters())==self.versions
        assert all(not p.requires_grad for p in self.model.parameters())

    def setup_side(self,side):
        if side in self.geometry:return self.geometry[side]
        prefix,_=text_parts(dict(instruction='fixture',candidates=['a','b']),self.c.get('prompt_style','short'))
        tokens=self.tokenizer.encode(prefix,add_special_tokens=False);image=self.model.image_token_id;assert tokens.count(image)==1
        at=tokens.index(image);nv=(side//32)**2
        expanded=tokens[:at]+[image]*nv+tokens[at+1:];grid=torch.tensor([[1,side//16,side//16]],dtype=torch.long)
        positions={}
        for size in self.c['buckets']:
            if size<=len(expanded):continue
            ids=torch.tensor([expanded+[0]*(size-len(expanded))]);p,_=self.model.vl_model.get_rope_index(ids,mm_token_type_ids=(ids==image).int(),image_grid_thw=grid)
            positions[size]=p.cuda()
        geo=self.vg.geo(grid.cuda())
        self.geometry[side]=dict(prefix=np.array(expanded,dtype=np.int64),image_start=at,nv=nv,vi=torch.arange(at,at+nv,device='cuda'),grid=grid,positions=positions,geo=geo)
        return self.geometry[side]

    def prepare(self,row,side=None):
        side=self.side if side is None else side;meta=self.setup_side(side)
        data=Path(row['image_path']).read_bytes()
        if data.startswith(b'\xff\xd8'):
            import imagecodecs
            rgb=imagecodecs.jpeg8_decode(data,outcolorspace='RGB')
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):
            import pyspng
            rgb=pyspng.load(data)
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:
                with Image.open(io.BytesIO(data)) as im:rgb=np.array(im.convert('RGB'))
        else:
            with Image.open(io.BytesIO(data)) as im:rgb=np.array(im.convert('RGB'))
        rgb=cv2.resize(rgb,(side,side),interpolation=cv2.INTER_AREA)
        prefix,suffix=text_parts(row,self.c.get('prompt_style','short'));ids=self.tokenizer.backend_tokenizer.encode(suffix,add_special_tokens=False).ids
        full=np.concatenate((meta['prefix'],np.asarray(ids,dtype=np.int64)));assert len(full)<=self.max_tokens,'No truncation'
        return dict(rgb=rgb,ids=full,pick=len(full)-1,k=len(row['candidates']),side=side,prompt=prefix+suffix)

    def native(self,row):
        with Image.open(row['image_path']) as im:
            g=self.model.prepare_group(im.convert('RGB'),[dict(instruction=row['instruction'],candidates=row['candidates'],qtype='choice')])
        z=self.model.run_independent(g)['lm_option_logits'][0,:len(row['candidates'])].float()
        return z.cpu().tolist()

    def reference(self,g):
        # Official HF on exactly the fast prompt/pixels; independent core oracle.
        p=self.model.processor(text=[g['prompt']],images=[Image.fromarray(g['rgb'])],do_resize=False,return_tensors='pt')
        assert p['input_ids'][0].tolist()==g['ids'].tolist()
        e={k:v.cuda().to(torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in p.items()}
        return self.model.backbone(**e,use_cache=False,logits_to_keep=1).logits[0,-1,self.option_ids[:g['k']]].float(),p

    def args(self,g,padded=True):
        n=len(g['ids']);b=next(v for v in self.c['buckets'] if v>=n) if padded else n
        ids=torch.zeros((1,b),device='cuda',dtype=torch.long);ids[0,:n].copy_(torch.from_numpy(g['ids']))
        meta=self.setup_side(g['side'])
        if padded:pos=meta['positions'][b]
        else:
            raw=torch.from_numpy(g['ids'])[None];pos,_=self.model.vl_model.get_rope_index(raw,mm_token_type_ids=(raw==self.model.image_token_id).int(),image_grid_thw=meta['grid']);pos=pos.cuda()
        return torch.from_numpy(g['rgb']).cuda(),ids,torch.tensor([g['pick']],device='cuda'),pos

    def vision(self,rgb,side):
        pixels=torch.empty(((side//16)**2,1536),device='cuda',dtype=torch.bfloat16)
        pack_rgb[(triton.cdiv(pixels.numel(),1024),)](rgb,pixels,side,pixels.numel(),1024,enable_fp_fusion=False)
        return self.vg.core(pixels,self.setup_side(side)['geo'])

    def prefix(self,args,side,indices=None,bf16=False,return_prefix=False):
        rgb,ids,pick,pos=args;meta=self.setup_side(side);visual,deep=self.vision(rgb,side)
        h=self.lm.embed_tokens(ids).index_copy(1,meta['vi'],visual[None]);rope=self.lm.rotary_emb(h,pos)
        indices=self.indices if indices is None else indices
        if bf16:
            for j in indices:
                h=self.lm.layers[j](h,attention_mask=None,position_embeddings=rope,position_ids=pos[0],cache_position=torch.arange(ids.shape[1],device='cuda'),use_cache=False)
                if j<len(deep):h[0].index_add_(0,meta['vi'],deep[j])
            return F.linear(self.lm.norm(h.index_select(1,pick)),self.head)[0,0].float()
        normalized=fast_norm(h,self.lm.layers[indices[0]].input_layernorm)
        for offset,j in enumerate(indices):
            if j==self.final:
                self.last_prefix=h
                if return_prefix:return h,rope,pick
                return self.tail(h,rope,pick)
            layer=self.lm.layers[j];attn=self.attention(j,normalized,rope)
            residual,post=add_rms(h,attn,None,layer.post_attention_layernorm)
            value=self.mlp(j,post)
            injection=deep[j] if j<len(deep) else None
            h,normalized=add_rms(residual,value,injection,self.lm.layers[indices[offset+1]].input_layernorm,meta['image_start'],meta['nv'])
        raise AssertionError('Final layer must be retained')

    def attention(self,j,x,rope):
        layer=self.lm.layers[j];a=layer.self_attn;pack=self.quant[j]
        q,k,v=Linear.apply(x,pack['qkv']).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],-1)
        shape=(*x.shape[:-1],-1,a.head_dim)
        q,k=qk_pair(q.view(shape),k.view(shape),a.q_norm,a.k_norm,*rope,4,4,False)
        v=v.view(shape).transpose(1,2)
        out=attention_direct(q,k,v,a.scaling,16,64,4,triangular=True)
        return Linear.apply(out.reshape(*x.shape[:-1],-1),pack['output'])

    def mlp(self,j,x):
        pack=self.quant[j];g,u=Linear.apply(x,pack['gateup']).chunk(2,-1)
        return Linear.apply(swiglu(g,u,512,4),pack['down'])

    def tail(self,h,rope,pick,repair=True):
        if repair:h=self.repair(h)
        if torch.is_grad_enabled() and h.requires_grad:
            return ExactTail.apply(h,*rope,pick,self)
        return self.tail_core(h,rope,pick)

    def tail_core(self,h,rope,pick):
        layer=self.lm.layers[self.final];residual,post=add_rms(h,self.attention(self.final,fast_norm(h,layer.input_layernorm),rope),None,layer.post_attention_layernorm)
        chosen=residual[0].index_select(0,pick)+self.mlp(self.final,post[0].index_select(0,pick))
        return lm_readout(chosen.contiguous(),self.lm.norm,self.head)[0].float()

    def pack_release(self,profile):
        self.frozen();self.quant=[];bank=nn.ParameterList();released=[]
        for j,layer in enumerate(self.lm.layers):
            a,m=layer.self_attn,layer.mlp
            weights=dict(qkv=torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]),output=a.o_proj.weight,gateup=torch.cat([m.gate_proj.weight,m.up_proj.weight]),down=m.down_proj.weight)
            pack={}
            for role,w in weights.items():
                op=StaticFP8Linear(w,448/profile[f'{j}:{role}']);pack[role]=op;bank.append(nn.Parameter(op.weight,requires_grad=False))
            self.quant.append(pack);del weights,w
            if not self.c.get('retain_bf16',True):
                for mod in [a.q_proj,a.k_proj,a.v_proj,a.o_proj,m.gate_proj,m.up_proj,m.down_proj]:
                    released.append(list(mod.weight.shape));mod.weight=nn.Parameter(torch.empty_like(mod.weight,device='meta'),requires_grad=False)
            gc.collect();torch.cuda.empty_cache()
            if j%8==0:print('PACK_RELEASE',j,torch.cuda.memory_allocated()/2**30,flush=True)
        self.model.add_module('execution_fp8',bank);self.versions=tuple(p._version for p in self.model.parameters())
        assert all(p.is_cuda or p.is_meta for p in self.model.parameters())
        return dict(released_linear_shapes=released,resident_cuda_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters() if p.is_cuda),canonical_disk_weights_unchanged=True)

    def configure(self,side,depth):
        self.cache.clear();self.side=side;self.indices=self.schedule(depth);self.setup_side(side)
        self.repair=Repair(self.width,rank=32,seed=731).requires_grad_(False)
        gc.collect();torch.cuda.empty_cache()

    def schedule(self,depth):
        plan=getattr(self,'layer_plan',None) or self.c.get('indices_by_depth',{})
        indices=plan.get(str(depth),list(range(depth-1))+[self.final])
        assert len(indices)==depth and indices==sorted(set(indices)) and indices[-1]==self.final
        return indices

    def graph(self,g,trace=False):
        n=len(g['ids']);bucket=next(v for v in self.c['buckets'] if v>=n)
        key=(g['side'],bucket);captured=key not in self.cache
        if captured:
            args=self.args(g)
            for _ in range(3):self.prefix(args,g['side'])
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):z=self.prefix(args,g['side'])
            self.cache[key]=(args,graph,z,self.last_prefix)
        static,graph,z,prefix=self.cache[key]
        # Refill graph inputs directly. Geometry is fixed by side/bucket;
        # image and complete text are fresh for every request.
        static[0].copy_(torch.from_numpy(g['rgb']),non_blocking=True)
        static[1].zero_();static[1][0,:n].copy_(torch.from_numpy(g['ids']),non_blocking=True)
        static[2].fill_(g['pick'])
        graph.replay()
        return (z[:g['k']],captured,prefix) if trace else (z[:g['k']],captured)


class ExactTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,cos,sin,pick,rt):
        ctx.save_for_backward(h,cos,sin,pick);ctx.rt=rt
        return rt.tail_core(h,(cos,sin),pick)

    @staticmethod
    def backward(ctx,gradient):
        h,cos,sin,pick=ctx.saved_tensors;rt=ctx.rt
        with torch.enable_grad():
            x=h.detach().requires_grad_(True);z=surrogate(rt,x,(cos,sin),pick)
            dh,=torch.autograd.grad(z,x,gradient)
        return dh,None,None,None,None


def surrogate(rt,h,rope,pick):
    from fastforward_accel._legacy.recovery_adapter import rms
    from torch.nn.attention import sdpa_kernel,SDPBackend
    layer=rt.lm.layers[rt.final];a=layer.self_attn;pack=rt.quant[rt.final]
    z=FP8STE.apply(rms(h,layer.input_layernorm),pack['qkv'])
    q,k,v=z.split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],-1)
    def rotary(x,norm):
        x=rms(x.view(*h.shape[:-1],-1,a.head_dim),norm).transpose(1,2);c,s=rope;c=c[:,None];s=s[:,None]
        left,right=x.chunk(2,-1)
        return (x*c).to(x.dtype)+(torch.cat((-right,left),-1)*s).to(x.dtype)
    q=rotary(q,a.q_norm);k=rotary(k,a.k_norm);v=v.view(*h.shape[:-1],-1,a.head_dim).transpose(1,2)
    with sdpa_kernel(SDPBackend.MATH):out=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=a.scaling,enable_gqa=True)
    residual=h+FP8STE.apply(out.transpose(1,2).reshape(*h.shape[:-1],-1),pack['output'])
    post=rms(residual,layer.post_attention_layernorm)[0].index_select(0,pick)
    g,u=FP8STE.apply(post,pack['gateup']).chunk(2,-1)
    chosen=residual[0].index_select(0,pick)+FP8STE.apply(F.silu(g)*u,pack['down'])
    return F.linear(rms(chosen,rt.lm.norm),rt.head)[0].float()
