"""Same BF16 rounding as fused_norm, with runtime token count for vLLM compile."""
import torch,triton
import triton.language as tl
@triton.jit
def _square(X,S,N,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);x=tl.load(X+i,i<N,other=0).to(tl.float32);tl.store(S+i,x*x,i<N)
@triton.jit
def _finish(X,V,W,Y,N,H:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N;x=tl.load(X+i,valid,other=0).to(tl.float32);v=tl.load(V+i//H,valid,other=0);w=tl.load(W+i%H,valid,other=0).to(tl.float32);z=(x*tl.rsqrt(v+EPS)).to(Y.dtype.element_ty).to(tl.float32);tl.store(Y+i,z*w,valid)
def fused_norm(x,module):
 assert x.is_contiguous() and x.dtype==module.weight.dtype==torch.bfloat16
 n=x.numel();square=torch.empty_like(x,dtype=torch.float32);_square[(triton.cdiv(n,1024),)](x,square,n,1024,enable_fp_fusion=False);variance=square.mean(-1,keepdim=True);out=torch.empty_like(x);_finish[(triton.cdiv(n,1024),)](x,variance,module.weight,out,n,x.shape[-1],module.variance_epsilon,1024,enable_fp_fusion=False);return out
