"""Private single-pass kernels; checkpoint weights and shared libraries stay read-only."""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms(A,B,D,W,R,Y,H:tl.constexpr,EPS:tl.constexpr,
             START:tl.constexpr,NV:tl.constexpr,DEEP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK);valid=i<H
    a=tl.load(A+row*H+i,valid,other=0).to(tl.float32)
    b=tl.load(B+row*H+i,valid,other=0).to(tl.float32)
    x=(a+b).to(R.dtype.element_ty).to(tl.float32)
    if DEEP:
        d=tl.load(D+(row-START)*H+i,valid & (row>=START) & (row<START+NV),other=0).to(tl.float32)
        x=(x+d).to(R.dtype.element_ty).to(tl.float32)
    variance=tl.sum(x*x,0)/H
    w=tl.load(W+i,valid,other=0).to(tl.float32)
    normalized=(x*tl.rsqrt(variance+EPS)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(R+row*H+i,x,valid)
    tl.store(Y+row*H+i,normalized*w,valid)


def add_rms(a,b,deep,module,start=0,nv=0):
    assert a.shape==b.shape and a.is_contiguous() and b.is_contiguous()
    assert a.dtype==b.dtype==module.weight.dtype==torch.bfloat16
    if deep is not None:assert deep.shape==(nv,a.shape[-1]) and deep.is_contiguous()
    residual=torch.empty_like(a);out=torch.empty_like(a);d=a.shape[-1]
    _add_rms[(a.numel()//d,)](a,b,deep if deep is not None else a,module.weight,residual,out,
        d,module.variance_epsilon,start,nv,deep is not None,triton.next_power_of_2(d),
        num_warps=4 if d<=2048 else 8,enable_fp_fusion=False)
    return residual,out


@triton.jit
def _norm_rope(X,W,C,S,Y,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
               STRIDE:tl.constexpr,EPS:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,
               ST:tl.constexpr,SD:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);token=row//H;head=row%H;i=tl.arange(0,BLOCK)
    ok=i<D;other=tl.where(i<D//2,i+D//2,i-D//2)
    x=tl.load(X+token*STRIDE+head*D+i,ok,other=0).to(tl.float32)
    r=tl.load(X+token*STRIDE+head*D+other,ok,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    w=tl.load(W+i,ok,other=0).to(tl.float32)
    wr=tl.load(W+other,ok,other=0).to(tl.float32)
    x=((x*inv).to(Y.dtype.element_ty).to(tl.float32)*w).to(Y.dtype.element_ty).to(tl.float32)
    r=((r*inv).to(Y.dtype.element_ty).to(tl.float32)*wr).to(Y.dtype.element_ty).to(tl.float32)
    r=tl.where(i<D//2,-r,r)
    c=tl.load(C+token*CT+i*CD,ok,other=0).to(tl.float32)
    s=tl.load(S+token*ST+i*SD,ok,other=0).to(tl.float32)
    left=(x*c).to(Y.dtype.element_ty).to(tl.float32)
    right=(r*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+(head*T+token)*D+i,left+right,ok)


def norm_rope(x,module,cos,sin):
    assert x.ndim==4 and x.shape[0]==1 and x.stride(-1)==1
    _,t,h,d=x.shape
    y=torch.empty((1,h,t,d),device=x.device,dtype=x.dtype)
    _norm_rope[(t*h,)](x,module.weight,cos,sin,y,t,h,d,x.stride(1),module.variance_epsilon,
        cos.stride(1),cos.stride(2),sin.stride(1),sin.stride(2),triton.next_power_of_2(d),
        num_warps=4,enable_fp_fusion=False)
    return y


@triton.jit
def _rope_rows(X,W,C,S,Y,base,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
               STRIDE:tl.constexpr,EPS:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,
               ST:tl.constexpr,SD:tl.constexpr,ROWS:tl.constexpr,BLOCK:tl.constexpr):
    row=base+tl.arange(0,ROWS);token=row//H;head=row%H;i=tl.arange(0,BLOCK)
    ok=(row[:,None]<T*H) & (i[None,:]<D)
    other=tl.where(i<D//2,i+D//2,i-D//2)
    at=token[:,None]*STRIDE+head[:,None]*D
    x=tl.load(X+at+i[None,:],ok,other=0).to(tl.float32)
    r=tl.load(X+at+other[None,:],ok,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS)
    w=tl.load(W+i,i<D,other=0).to(tl.float32)
    wr=tl.load(W+other,i<D,other=0).to(tl.float32)
    x=((x*inv[:,None]).to(Y.dtype.element_ty).to(tl.float32)*w[None,:]).to(Y.dtype.element_ty).to(tl.float32)
    r=((r*inv[:,None]).to(Y.dtype.element_ty).to(tl.float32)*wr[None,:]).to(Y.dtype.element_ty).to(tl.float32)
    r=tl.where(i[None,:]<D//2,-r,r)
    c=tl.load(C+token[:,None]*CT+i[None,:]*CD,ok,other=0).to(tl.float32)
    s=tl.load(S+token[:,None]*ST+i[None,:]*SD,ok,other=0).to(tl.float32)
    left=(x*c).to(Y.dtype.element_ty).to(tl.float32)
    right=(r*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+(head[:,None]*T+token[:,None])*D+i[None,:],left+right,ok)


@triton.jit
def _pair_rows(Q,K,WQ,WK,C,S,YQ,YK,T:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
               D:tl.constexpr,STRIDE:tl.constexpr,EQ:tl.constexpr,EK:tl.constexpr,
               CT:tl.constexpr,CD:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr,
               ROWS:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0);nq=tl.cdiv(T*HQ,ROWS)
    if pid<nq:
        _rope_rows(Q,WQ,C,S,YQ,pid*ROWS,T,HQ,D,STRIDE,EQ,CT,CD,ST,SD,ROWS,BLOCK)
    else:
        _rope_rows(K,WK,C,S,YK,(pid-nq)*ROWS,T,HK,D,STRIDE,EK,CT,CD,ST,SD,ROWS,BLOCK)


def norm_rope_pair(q,k,qn,kn,cos,sin,rows=4):
    assert q.ndim==k.ndim==4 and q.shape[0]==k.shape[0]==1
    assert q.shape[1]==k.shape[1] and q.shape[-1]==k.shape[-1]
    assert q.stride(1)==k.stride(1) and q.stride(-1)==k.stride(-1)==1
    _,t,hq,d=q.shape;hk=k.shape[2]
    yq=torch.empty((1,hq,t,d),device=q.device,dtype=q.dtype)
    yk=torch.empty((1,hk,t,d),device=k.device,dtype=k.dtype)
    _pair_rows[(triton.cdiv(t*hq,rows)+triton.cdiv(t*hk,rows),)](q,k,qn.weight,kn.weight,
        cos,sin,yq,yk,t,hq,hk,d,q.stride(1),qn.variance_epsilon,kn.variance_epsilon,
        cos.stride(1),cos.stride(2),sin.stride(1),sin.stride(2),rows,triton.next_power_of_2(d),
        num_warps=rows,enable_fp_fusion=False)
    return yq,yk


@triton.jit
def _add_layernorm(A,B,W,BIAS,R,Y,H:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK);ok=i<H
    a=tl.load(A+row*H+i,ok,other=0).to(tl.float32)
    b=tl.load(B+row*H+i,ok,other=0).to(tl.float32)
    x=(a+b).to(R.dtype.element_ty).to(tl.float32)
    mean=tl.sum(x,0)/H
    centered=tl.where(ok,x-mean,0.)
    inv=tl.rsqrt(tl.sum(centered*centered,0)/H+EPS)
    w=tl.load(W+i,ok,other=0).to(tl.float32)
    bias=tl.load(BIAS+i,ok,other=0).to(tl.float32)
    tl.store(R+row*H+i,x,ok)
    tl.store(Y+row*H+i,centered*inv*w+bias,ok)


def add_layernorm(a,b,module):
    assert a.shape==b.shape and a.is_contiguous() and b.is_contiguous()
    assert a.dtype==b.dtype==module.weight.dtype==torch.bfloat16
    residual=torch.empty_like(a);out=torch.empty_like(a);d=a.shape[-1]
    _add_layernorm[(a.numel()//d,)](a,b,module.weight,module.bias,residual,out,d,module.eps,
        triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return residual,out
