"""Fuse Q/K RMSNorm finish and rotary; preserve torch mean and BF16 boundaries."""
import torch,triton
import triton.language as tl
from shunong_accel._legacy.serving_dynamic_norm import _square
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS

@triton.jit
def finish_rotary(X,V,W,C,S,Y,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,EPS:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N;col=i%D;token=(i//D)%T;head=i//(D*T);row=token*H+head;other=tl.where(col<D//2,col+D//2,col-D//2)
    x=tl.load(X+row*D+col,ok,other=0).to(tl.float32);r=tl.load(X+row*D+other,ok,other=0).to(tl.float32)
    var=tl.load(V+row,ok,other=0);inv=tl.rsqrt(var+EPS);w=tl.load(W+col,ok,other=0).to(tl.float32);wr=tl.load(W+other,ok,other=0).to(tl.float32)
    x=((x*inv).to(Y.dtype.element_ty).to(tl.float32)*w).to(Y.dtype.element_ty).to(tl.float32)
    r=((r*inv).to(Y.dtype.element_ty).to(tl.float32)*wr).to(Y.dtype.element_ty).to(tl.float32);r=tl.where(col<D//2,-r,r)
    c=tl.load(C+token*CT+col*CD,ok,other=0).to(tl.float32);s=tl.load(S+token*ST+col*SD,ok,other=0).to(tl.float32)
    left=(x*c).to(Y.dtype.element_ty).to(tl.float32);right=(r*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,left+right,ok)

def norm_rotary(x,module,cos,sin):
    assert x.ndim==4 and x.shape[0]==cos.shape[0]==sin.shape[0]==1 and x.is_contiguous()
    assert x.dtype==module.weight.dtype==cos.dtype==sin.dtype==torch.bfloat16
    _,t,h,d=x.shape;n=x.numel();square=torch.empty_like(x,dtype=torch.float32)
    _square[(triton.cdiv(n,1024),)](x,square,n,1024,enable_fp_fusion=False)
    variance=square.mean(-1,keepdim=True);out=torch.empty((1,h,t,d),device=x.device,dtype=x.dtype)
    finish_rotary[(triton.cdiv(n,512),)](x,variance,module.weight,cos,sin,out,n,t,h,d,module.variance_epsilon,cos.stride(1),cos.stride(2),sin.stride(1),sin.stride(2),512,enable_fp_fusion=False)
    return out

def attention(a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
    shape=hidden_states.shape[:-1];head_shape=(*shape,-1,a.head_dim)
    q,k,v=torch.nn.functional.linear(hidden_states,w).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],dim=-1)
    q=norm_rotary(q.contiguous().view(head_shape),a.q_norm,*position_embeddings)
    k=norm_rotary(k.contiguous().view(head_shape),a.k_norm,*position_embeddings)
    v=v.view(head_shape).transpose(1,2)
    assert kwargs.get('past_key_values') is None and a.config._attn_implementation=='sdpa'
    out,_=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,v,attention_mask,dropout=0.,scaling=a.scaling)
    return a.o_proj(out.reshape(*shape,-1).contiguous()),None
