"""Equivalent activation cast with a runtime scale, avoiding layer specialization."""
import torch
import triton
import triton.language as tl
from shunong_accel._legacy.limit_kernels import StaticFP8Linear,_static_fp8


@triton.jit
def _runtime_scale(X,Q,M:tl.constexpr,K:tl.constexpr,INV,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.load(X+i,i<M*K,0).to(tl.float32)*INV
    tl.store(Q+i,tl.minimum(448.,tl.maximum(-448.,value)),i<tl.cdiv(M,16)*16*K)


def quantize_runtime(x,op):
    x=x.reshape(-1,x.shape[-1]).contiguous();m,k=x.shape
    q=torch.empty((triton.cdiv(m,16)*16,k),device=x.device,dtype=torch.float8_e4m3fn)
    _runtime_scale[(triton.cdiv(q.numel(),1024),)](x,q,m,k,op.inverse_scale,1024,num_warps=4)
    return q,m


class RuntimeScaleLinear(StaticFP8Linear):
    def __call__(self,x):
        q,m=quantize_runtime(x,self)
        return torch._scaled_mm(q,self.weight.T,scale_a=self.input_scale,scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,use_fast_accum=getattr(self,'fast_accum',True))[:m]


def parity(op,x):
    flat=x.reshape(-1,x.shape[-1]).contiguous();m,k=flat.shape
    q,_=quantize_runtime(x,op);old=torch.empty_like(q)
    _static_fp8[(triton.cdiv(old.numel(),1024),)](flat,old,m,k,op.inverse_scale,1024,num_warps=4)
    exact=torch.equal(q.view(torch.uint8),old.view(torch.uint8))
    new=op(x);previous=StaticFP8Linear.__call__(op,x)
    result=dict(shape=list(flat.shape),inverse_scale=op.inverse_scale,quant_bytes_exact=exact,output_exact=torch.equal(new,previous))
    assert exact and result['output_exact'],result
    return result


def model_parity(rt,sample):
    captured={};original=[]
    class Observe:
        def __init__(self,op,key):self.op=op;self.key=key
        def __getattr__(self,name):return getattr(self.op,name)
        def __call__(self,x):
            captured[self.key]=(self.op,x.detach().clone())
            return self.op(x)
    for j in [rt.depth//2,rt.final]:
        for role,op in list(rt.quant[j].items()):
            if j==rt.depth//2 or role in ['gateup','down']:
                original.append((j,role,op));rt.quant[j][role]=Observe(op,f'{j}:{role}')
    g=rt.prepare(sample);rt.prefix(rt.args(g),g['side'])
    for j,role,op in original:rt.quant[j][role]=op
    checks=[]
    profiles=getattr(rt,'profiles',{}) or {'anchor':None}
    for key,(op,x) in captured.items():
        inverse=op.inverse_scale;scale=op.input_scale.clone()
        for name,profile in profiles.items():
            if profile is not None:
                op.inverse_scale=448/profile[key];op.input_scale.fill_(profile[key]/448)
            checks.append(dict(role=key,variant=name,**parity(op,x)))
        op.inverse_scale=inverse;op.input_scale.copy_(scale)
    return checks
