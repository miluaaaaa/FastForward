"""Fuse RMSNorm pointwise work, preserving native FP32 mean and BF16 rounding."""
import torch
import triton
import triton.language as tl
from fastforward_accel._legacy.graph24_runtime import Graph24

@triton.jit
def _square(X,S,N:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);x=tl.load(X+i,i<N,other=0).to(tl.float32);tl.store(S+i,x*x,i<N)

@triton.jit
def _finish(X,V,W,Y,N:tl.constexpr,H:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N;x=tl.load(X+i,valid,other=0).to(tl.float32);v=tl.load(V+i//H,valid,other=0);w=tl.load(W+i%H,valid,other=0).to(tl.float32)
 # Native code rounds normalized values to BF16 BEFORE multiplying BF16 weight.
 normalized=(x*tl.rsqrt(v+EPS)).to(Y.dtype.element_ty).to(tl.float32);tl.store(Y+i,normalized*w,valid)

def fused_norm(x,module):
 assert x.is_contiguous() and x.dtype==module.weight.dtype==torch.bfloat16
 square=torch.empty_like(x,dtype=torch.float32);n=x.numel();_square[(triton.cdiv(n,1024),)](x,square,n,1024,enable_fp_fusion=False)
 variance=square.mean(-1,keepdim=True);out=torch.empty_like(x);_finish[(triton.cdiv(n,1024),)](x,variance,module.weight,out,n,x.shape[-1],module.variance_epsilon,1024,enable_fp_fusion=False);return out

class FusedGraph24(Graph24):
 def core(self,args):
  x,ds,pos,picks,mask,_=args;rope=self.m.vl_model.language_model.rotary_emb(x,pos);cache_position=torch.arange(x.shape[1],device='cuda');h=x
  for j,layer in enumerate(self.layers):
   residual=h;norm=fused_norm(h,layer.input_layernorm)
   attn,_=layer.self_attn(hidden_states=norm,attention_mask=None,position_ids=pos[0],past_key_values=None,use_cache=False,cache_position=cache_position,position_embeddings=rope)
   h=residual+attn;residual=h;h=residual+layer.mlp(fused_norm(h,layer.post_attention_layernorm))
   if j<len(ds):h=h+ds[j]
  chosen=h[0].index_select(0,picks);return self.head.score(chosen[8][None],chosen[:8][None],mask)
