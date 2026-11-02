"""Private row-scaled FP8 matrices; adapted from the prior pilot kernel.

No module forwards or original checkpoint tensors are replaced. Activation
scales are recomputed from each actual request, including zero padding rows.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M'])
def quantize_rows(X,Q,S,M,K:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,B)
    value=tl.load(X+row*K+cols,(row<M)&(cols<K),other=0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(value),0),1e-12)/448.
    quant=tl.minimum(tl.maximum(value/scale,-448.),448.)
    tl.store(Q+row*K+cols,quant,cols<K)
    tl.store(S+row,scale)


class FP8Linear:
    def __init__(self,weight):
        assert weight.is_cuda and weight.ndim==2 and weight.shape[0]%16==weight.shape[1]%16==0
        w=weight.detach().float()
        scale=w.abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/448.
        self.weight=(w/scale).clamp(-448,448).to(torch.float8_e4m3fn).contiguous()
        self.scale=scale.T.contiguous()

    def __call__(self,x):
        assert x.is_cuda and x.dtype==torch.bfloat16
        flat=x.reshape(-1,x.shape[-1]).contiguous();m,k=flat.shape
        pad=triton.cdiv(m,16)*16
        q=torch.empty((pad,k),device=x.device,dtype=torch.float8_e4m3fn)
        scales=torch.empty((pad,1),device=x.device,dtype=torch.float32)
        quantize_rows[(pad,)](flat,q,scales,m,k,triton.next_power_of_2(k),num_warps=8 if k>4096 else 4)
        z=torch._scaled_mm(q,self.weight.T,scale_a=scales,scale_b=self.scale,
                           out_dtype=torch.bfloat16,use_fast_accum=False)
        return z[:m].reshape(*x.shape[:-1],self.weight.shape[0])
