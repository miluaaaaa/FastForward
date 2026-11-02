"""Private kernels with explicit historical FP32/BF16 boundaries."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu(G,U,Y,N,I:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    offset=i//I*STRIDE+i%I
    g=tl.load(G+offset,ok,other=0).to(tl.float32)
    u=tl.load(U+offset,ok,other=0).to(tl.float32)
    activated=libdevice.div_rn(g,1.+libdevice.exp(-g)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,activated*u,ok)


def swiglu(gate,up,block=1024,warps=4):
    assert gate.shape==up.shape and gate.stride()==up.stride()
    assert gate.stride(-1)==1 and gate.dtype==up.dtype==torch.bfloat16
    y=torch.empty(gate.shape,device=gate.device,dtype=gate.dtype)
    _swiglu[(triton.cdiv(y.numel(),block),)](gate,up,y,y.numel(),gate.shape[-1],gate.stride(-2),block,num_warps=warps,enable_fp_fusion=False)
    return y


@triton.jit
def _square_tile(X,S,i,N,H:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr):
    row=i//D;col=i%D
    source=(row//H)*STRIDE+(row%H)*D+col
    x=tl.load(X+source,i<N,other=0).to(tl.float32)
    tl.store(S+i,x*x,i<N)


@triton.jit
def _square_strided(X,S,N,H:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    _square_tile(X,S,i,N,H,D,STRIDE)


@triton.jit
def _squares_joint(Q,K,SQ,SK,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0);qb=tl.cdiv(NQ,B)
    if pid<qb:
        i=pid*B+tl.arange(0,B)
        _square_tile(Q,SQ,i,NQ,HQ,D,STRIDE)
    else:
        i=(pid-qb)*B+tl.arange(0,B)
        _square_tile(K,SK,i,NK,HK,D,STRIDE)


@triton.jit
def _rope_tile(X,V,W,C,S,Y,i,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,EPS:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr):
    ok=i<N;col=i%D;token=(i//D)%T;head=i//(D*T);row=token*H+head
    other=tl.where(col<D//2,col+D//2,col-D//2)
    source=token*STRIDE+head*D
    x=tl.load(X+source+col,ok,other=0).to(tl.float32)
    r=tl.load(X+source+other,ok,other=0).to(tl.float32)
    var=tl.load(V+row,ok,other=0);inv=tl.rsqrt(var+EPS)
    w=tl.load(W+col,ok,other=0).to(tl.float32);wr=tl.load(W+other,ok,other=0).to(tl.float32)
    x=((x*inv).to(Y.dtype.element_ty).to(tl.float32)*w).to(Y.dtype.element_ty).to(tl.float32)
    r=((r*inv).to(Y.dtype.element_ty).to(tl.float32)*wr).to(Y.dtype.element_ty).to(tl.float32)
    r=tl.where(col<D//2,-r,r)
    c=tl.load(C+token*CT+col*CD,ok,other=0).to(tl.float32)
    s=tl.load(S+token*ST+col*SD,ok,other=0).to(tl.float32)
    left=(x*c).to(Y.dtype.element_ty).to(tl.float32)
    right=(r*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,left+right,ok)


@triton.jit
def _rope_strided(X,V,W,C,S,Y,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,EPS:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    _rope_tile(X,V,W,C,S,Y,i,N,T,H,D,STRIDE,EPS,CT,CD,ST,SD)


@triton.jit
def _rope_joint(Q,K,VQ,VK,WQ,WK,C,S,YQ,YK,NQ:tl.constexpr,NK:tl.constexpr,T:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,EQ:tl.constexpr,EK:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0);qb=tl.cdiv(NQ,B)
    if pid<qb:
        i=pid*B+tl.arange(0,B)
        _rope_tile(Q,VQ,WQ,C,S,YQ,i,NQ,T,HQ,D,STRIDE,EQ,CT,CD,ST,SD)
    else:
        i=(pid-qb)*B+tl.arange(0,B)
        _rope_tile(K,VK,WK,C,S,YK,i,NK,T,HK,D,STRIDE,EK,CT,CD,ST,SD)


def norm_rope_pair(q,k,qn,kn,cos,sin,joint=False,block=512):
    assert q.ndim==k.ndim==4 and q.shape[0]==k.shape[0]==1
    _,t,hq,d=q.shape;hk=k.shape[2]
    assert q.stride(1)==k.stride(1) and q.stride(-1)==k.stride(-1)==1
    sq=torch.empty(q.shape,device=q.device,dtype=torch.float32)
    sk=torch.empty(k.shape,device=k.device,dtype=torch.float32)
    nq,nk=q.numel(),k.numel();stride=q.stride(1)
    if joint:
        _squares_joint[(triton.cdiv(nq,1024)+triton.cdiv(nk,1024),)](q,k,sq,sk,nq,nk,hq,hk,d,stride,1024,enable_fp_fusion=False)
    else:
        _square_strided[(triton.cdiv(nq,1024),)](q,sq,nq,hq,d,stride,1024,enable_fp_fusion=False)
        _square_strided[(triton.cdiv(nk,1024),)](k,sk,nk,hk,d,stride,1024,enable_fp_fusion=False)
    vq=sq.mean(-1,keepdim=True);vk=sk.mean(-1,keepdim=True)
    yq=torch.empty((1,hq,t,d),device=q.device,dtype=q.dtype)
    yk=torch.empty((1,hk,t,d),device=k.device,dtype=k.dtype)
    strides=(cos.stride(1),cos.stride(2),sin.stride(1),sin.stride(2))
    if joint:
        _rope_joint[(triton.cdiv(nq,block)+triton.cdiv(nk,block),)](q,k,vq,vk,qn.weight,kn.weight,cos,sin,yq,yk,nq,nk,t,hq,hk,d,stride,qn.variance_epsilon,kn.variance_epsilon,*strides,block,enable_fp_fusion=False)
    else:
        _rope_strided[(triton.cdiv(nq,block),)](q,vq,qn.weight,cos,sin,yq,nq,t,hq,d,stride,qn.variance_epsilon,*strides,block,enable_fp_fusion=False)
        _rope_strided[(triton.cdiv(nk,block),)](k,vk,kn.weight,cos,sin,yk,nk,t,hk,d,stride,kn.variance_epsilon,*strides,block,enable_fp_fusion=False)
    return yq,yk


@triton.jit
def _vision_rope(Q,K,C,S,YQ,YK,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    token=i//(H*D);head=i//D%H;col=i%D
    other=tl.where(col<D//2,col+D//2,col-D//2)
    at=token*STRIDE+head*D
    q=tl.load(Q+at+col,ok,other=0).to(tl.float32)
    qr=tl.load(Q+at+other,ok,other=0).to(tl.float32)
    k=tl.load(K+at+col,ok,other=0).to(tl.float32)
    kr=tl.load(K+at+other,ok,other=0).to(tl.float32)
    c=tl.load(C+token*D+col,ok,other=0).to(tl.float32)
    s=tl.load(S+token*D+col,ok,other=0).to(tl.float32)
    qr=tl.where(col<D//2,-qr,qr);kr=tl.where(col<D//2,-kr,kr)
    tl.store(YQ+i,q*c+qr*s,ok);tl.store(YK+i,k*c+kr*s,ok)


def vision_rope(q,k,cos,sin,block=512):
    assert q.shape==k.shape and q.stride()==k.stride()
    assert q.ndim==3 and q.stride(-1)==1 and cos.is_contiguous() and sin.is_contiguous()
    n,h,d=q.shape
    yq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    yk=torch.empty_like(yq)
    _vision_rope[(triton.cdiv(q.numel(),block),)](q,k,cos,sin,yq,yk,q.numel(),h,d,q.stride(0),block,enable_fp_fusion=False)
    return yq,yk
