"""Experimental rectangular inputs; installed 0.2.3 and prior releases stay intact."""
import io
import gc
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import triton
import triton.language as tl
from fastforward_accel.scaled_qwen_runtime import Runtime as Base, NativeNormVision
from fastforward_accel._legacy.sub10_runtime import text_parts
from .depth_geometry import canvas, fit
from .depth_fp8 import RuntimeScaleLinear


@triton.jit
def pack_rectangle(X, Y, WIDTH: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N;p=i//1536;d=i%1536
    c=d//512;y0=(d%256)//16;x0=d%16;gw=WIDTH//16
    gy=p//(gw//2*4);gx=(p//4)%(gw//2);my=(p%4)//2;mx=p%2
    y=gy*32+my*16+y0;x=gx*32+mx*16+x0
    raw=tl.load(X+(y*WIDTH+x)*3+c,valid,other=0).to(tl.float32)
    tl.store(Y+i,(raw-127.5)/127.5,valid)


class Runtime(Base):
    def __init__(self, config):
        super().__init__(config)
        self.variant='square256'
        self.vg=NativeNormVision(self.model)
        self.profiles={}

    def setup_side(self, side):
        side=tuple(side)
        if side in self.geometry:return self.geometry[side]
        height,width=side
        prefix,_=text_parts(dict(instruction='fixture',candidates=['a','b']),self.c.get('prompt_style','short'))
        tokens=self.tokenizer.encode(prefix,add_special_tokens=False);image=self.model.image_token_id
        assert tokens.count(image)==1
        at=tokens.index(image);nv=(height//32)*(width//32)
        expanded=tokens[:at]+[image]*nv+tokens[at+1:]
        grid=torch.tensor([[1,height//16,width//16]],dtype=torch.long);positions={}
        for size in self.c['buckets']:
            if size<=len(expanded):continue
            ids=torch.tensor([expanded+[0]*(size-len(expanded))])
            p,_=self.model.vl_model.get_rope_index(ids,mm_token_type_ids=(ids==image).int(),image_grid_thw=grid)
            positions[size]=p.cuda()
        self.geometry[side]=dict(prefix=np.array(expanded,dtype=np.int64),image_start=at,nv=nv,
            vi=torch.arange(at,at+nv,device='cuda'),grid=grid,positions=positions,geo=self.vg.geo(grid.cuda()))
        return self.geometry[side]

    def prepare(self,row,side=None):
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
        height,width=rgb.shape[:2];shape=canvas(height,width,self.variant)
        if self.variant.startswith('aspect'):
            nh,nw,top,left=fit(height,width,shape)
            result=np.full((*shape,3),128,dtype=np.uint8)
            result[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
        else:result=cv2.resize(rgb,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
        meta=self.setup_side(shape);prefix,suffix=text_parts(row,self.c.get('prompt_style','short'))
        ids=self.tokenizer.backend_tokenizer.encode(suffix,add_special_tokens=False).ids
        full=np.concatenate((meta['prefix'],np.asarray(ids,dtype=np.int64)))
        assert len(full)<=self.max_tokens,'No text truncation'
        return dict(rgb=result,ids=full,pick=len(full)-1,k=len(row['candidates']),side=shape,prompt=prefix+suffix)

    def vision(self,rgb,side):
        height,width=side
        pixels=torch.empty(((height//16)*(width//16),1536),device='cuda',dtype=torch.bfloat16)
        pack_rectangle[(triton.cdiv(pixels.numel(),1024),)](rgb,pixels,width,pixels.numel(),1024,enable_fp_fusion=False)
        return self.vg.core(pixels,self.setup_side(side)['geo'])

    def select(self,variant):
        self.variant=variant
        if self.quant is not None:
            for j,pack in enumerate(self.quant):
                for role,op in pack.items():
                    bound=self.profiles[variant][f'{j}:{role}']
                    op.inverse_scale=448/bound
                    op.input_scale.fill_(bound/448)

    def clear_graphs(self):
        self.cache.clear()
        if hasattr(self,'last_prefix'):del self.last_prefix
        import gc
        gc.collect();torch.cuda.empty_cache()

    def pack_release(self,profile):
        self.frozen();self.quant=[];bank=nn.ParameterList();released=[]
        for j,layer in enumerate(self.lm.layers):
            a,m=layer.self_attn,layer.mlp
            weights=dict(qkv=torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]),output=a.o_proj.weight,
                gateup=torch.cat([m.gate_proj.weight,m.up_proj.weight]),down=m.down_proj.weight)
            pack={}
            for role,w in weights.items():
                op=RuntimeScaleLinear(w,448/profile[f'{j}:{role}']);pack[role]=op;bank.append(nn.Parameter(op.weight,requires_grad=False))
            self.quant.append(pack);del weights,w
            if not self.c.get('retain_bf16',True):
                for mod in [a.q_proj,a.k_proj,a.v_proj,a.o_proj,m.gate_proj,m.up_proj,m.down_proj]:
                    released.append(list(mod.weight.shape));mod.weight=nn.Parameter(torch.empty_like(mod.weight,device='meta'),requires_grad=False)
            gc.collect();torch.cuda.empty_cache()
            if j%8==0:print('PACK_RELEASE',j,torch.cuda.memory_allocated()/2**30,flush=True)
        self.model.add_module('execution_fp8',bank);self.versions=tuple(p._version for p in self.model.parameters())
        assert all(p.is_cuda or p.is_meta for p in self.model.parameters())
        return dict(released_linear_shapes=released,runtime_activation_scale=True,
            resident_cuda_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters() if p.is_cuda),canonical_disk_weights_unchanged=True)
