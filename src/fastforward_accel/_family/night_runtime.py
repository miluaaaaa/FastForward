"""Experimental whole-family FP8/W4A16 execution; original site is immutable."""
import gc,types
import torch
from torch import nn
from torch.nn import functional as F
from .depth_runtime import Runtime as Rectangular
from .depth_fp8 import RuntimeScaleLinear
from fastforward_accel.scaled_qwen_runtime import Linear
from fastforward_accel._legacy.sub10_runtime import fast_norm
from fastforward_accel._legacy.broad_kernels import qk_pair
from fastforward_accel._legacy.limit_kernels import attention_direct,lm_readout
from fastforward_accel._legacy.lower_kernels import add_rms


class Int4:
    def __init__(self,w,oracle=False):
        n,k=w.shape;assert n%8==0 and k%128==0
        self.weight=types.SimpleNamespace(shape=(n,k));self.group=32
        x=w.float().reshape(n,-1,self.group)
        low=x.amin(-1,keepdim=True);high=x.amax(-1,keepdim=True)
        scale=((high-low)/15).clamp_min(1e-6)
        q=((x-low)/scale).round().clamp(0,15).to(torch.uint8).reshape(n,k)
        zeros=low+scale*8
        self.scales=torch.stack((scale.squeeze(-1),zeros.squeeze(-1)),dim=-1).transpose(0,1).contiguous().bfloat16()
        packed=(q[:,0::2]<<4)|q[:,1::2]
        self.packed=torch.ops.aten._convert_weight_to_int4pack(packed.contiguous(),8)
        self.oracle=None;self.operator_check=None
        if oracle:
            s,z=self.scales.transpose(0,1).float().unbind(-1)
            self.oracle=((q.reshape(n,-1,self.group).float()-8)*s[...,None]+z[...,None]).reshape(n,k)
    def __call__(self,x):
        flat=x.reshape(-1,x.shape[-1]).contiguous()
        y=torch.ops.aten._weight_int4pack_mm(flat,self.packed,self.group,self.scales)
        if self.oracle is not None:
            # A real model activation, not a synthetic load or model fallback.
            reference=F.linear(flat[:1].float(),self.oracle)
            delta=y[:1].float()-reference
            rms=float(delta.square().mean().sqrt()/(reference.square().mean().sqrt()+1e-12))
            self.operator_check=dict(passed=rms<=.01,relative_rms=rms,max_abs=float(delta.abs().max()),shape=list(flat.shape))
            assert self.operator_check['passed'],self.operator_check
            self.oracle=None
        return y


class Runtime(Rectangular):
    def __init__(self,c):
        super().__init__(c);self.mode='anchor';self.banks={};self.staging={}
    def select(self,variant):
        self.variant=variant
        if self.banks:
            for j,pack in enumerate(self.banks['fp8']):
                for role,op in pack.items():
                    bound=self.profiles[variant][f'{j}:{role}']
                    op.inverse_scale=448/bound;op.input_scale.fill_(bound/448)
    def set_mode(self,mode):
        assert mode in ['anchor','fp8','int4','precise','int4_half','int4_ffn'];self.clear_graphs();self.mode=mode
        if self.banks:
            for pack in self.banks['fp8']:
                for op in pack.values():op.fast_accum=mode!='precise'
            if mode in ['int4_half','int4_ffn']:
                self.quant=[]
                for j,pack in enumerate(self.banks['fp8']):
                    interior=3<=j<self.final-3
                    self.quant.append({role:self.banks['int4'][j][role] if interior and (j%2==0 if mode=='int4_half' else role in ['gateup','down']) else op for role,op in pack.items()})
            else:self.quant=self.banks['int4' if mode=='int4' else 'fp8']
    def clear_graphs(self):
        torch.cuda.synchronize();super().clear_graphs();self.staging.clear()
    def pack_release(self,profile):
        self.frozen();banks={'fp8':[],'int4':[]};parameters=nn.ParameterList()
        for j,layer in enumerate(self.lm.layers):
            a,m=layer.self_attn,layer.mlp
            weights=dict(qkv=torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]),output=a.o_proj.weight,
                gateup=torch.cat([m.gate_proj.weight,m.up_proj.weight]),down=m.down_proj.weight)
            fp={};i4={}
            for role,w in weights.items():
                fp[role]=RuntimeScaleLinear(w,448/profile[f'{j}:{role}'])
                i4[role]=Int4(w,oracle=j==self.depth//2 and role=='qkv')
                for value in [fp[role].weight,i4[role].packed,i4[role].scales]:
                    parameters.append(nn.Parameter(value,requires_grad=False))
            banks['fp8'].append(fp);banks['int4'].append(i4);del weights,w
            for mod in [a.q_proj,a.k_proj,a.v_proj,a.o_proj,m.gate_proj,m.up_proj,m.down_proj]:
                mod.weight=nn.Parameter(torch.empty_like(mod.weight,device='meta'),requires_grad=False)
            gc.collect();torch.cuda.empty_cache()
            if j%8==0:print('PACK_RELEASE',j,torch.cuda.memory_allocated()/2**30,flush=True)
        self.banks=banks;self.quant=banks['fp8'];self.model.add_module('execution_nightly',parameters)
        self.versions=tuple(p._version for p in self.model.parameters())
        assert all(p.is_cuda or p.is_meta for p in self.model.parameters())
        assert torch.count_nonzero(self.repair.up)==0 and torch.count_nonzero(self.repair.bias)==0
        return dict(fp8_bytes=sum(op.weight.numel() for p in banks['fp8'] for op in p.values()),
            int4_bytes=sum(op.packed.numel()*4+op.scales.numel()*2 for p in banks['int4'] for op in p.values()),
            resident_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters() if p.is_cuda),group=32)
    def tail(self,h,rope,pick,repair=True):
        if self.mode=='anchor':return super().tail(h,rope,pick,repair)
        # The zero initialized repair is proved identity before this path is used.
        layer=self.lm.layers[self.final];a=layer.self_attn;pack=self.quant[self.final]
        x=fast_norm(h,layer.input_layernorm)
        q,k,v=Linear.apply(x,pack['qkv']).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],-1)
        shape=(*x.shape[:-1],-1,a.head_dim)
        q,k=qk_pair(q.view(shape),k.view(shape),a.q_norm,a.k_norm,*rope,4,4,False)
        v=v.view(shape).transpose(1,2);query=q.index_select(2,pick)
        out=attention_direct(query,k,v,a.scaling,16,64,4,positions=pick)
        value=Linear.apply(out.reshape(1,1,-1),pack['output'])
        residual,post=add_rms(h.index_select(1,pick),value,None,layer.post_attention_layernorm)
        chosen=residual[0]+self.mlp(self.final,post[0])
        return lm_readout(chosen.contiguous(),self.lm.norm,self.head)[0].float()
    def graph(self,g,trace=False):
        if self.mode=='anchor':return super().graph(g,trace)
        n=len(g['ids']);bucket=next(v for v in self.c['buckets'] if v>=n);key=(g['side'],bucket)
        captured=key not in self.cache
        if captured:
            args=self.args(g)
            for _ in range(3):self.prefix(args,g['side'])
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):z=self.prefix(args,g['side'])
            self.cache[key]=(args,graph,z,self.last_prefix)
            rgb=torch.empty(g['rgb'].shape,dtype=torch.uint8,pin_memory=True)
            ids=torch.empty((1,bucket),dtype=torch.long,pin_memory=True)
            pick=torch.empty((1,),dtype=torch.long,pin_memory=True)
            event=torch.cuda.Event();event.record();self.staging[key]=(rgb,ids,pick,event)
        static,graph,z,prefix=self.cache[key];rgb,ids,pick,event=self.staging[key]
        event.synchronize()  # Only the preceding H2D must finish before host reuse.
        rgb.copy_(torch.from_numpy(g['rgb']));ids.zero_();ids[0,:n].copy_(torch.from_numpy(g['ids']));pick[0]=g['pick']
        static[0].copy_(rgb,non_blocking=True);static[1].copy_(ids,non_blocking=True);static[2].copy_(pick,non_blocking=True)
        event.record();graph.replay()
        return (z[:g['k']],captured,prefix) if trace else (z[:g['k']],captured)
