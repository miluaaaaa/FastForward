"""Private experimental kernels. No shared package or checkpoint mutation."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from fastforward_accel._legacy.lower_kernels import _add_rms, _add_layernorm, _pair_rows


def add_rms(a,b,deep,module,start=0,nv=0,warps=4,fusion=False):
    assert a.shape==b.shape and a.is_contiguous() and b.is_contiguous()
    assert a.dtype==b.dtype==module.weight.dtype==torch.bfloat16
    if deep is not None:assert deep.shape==(nv,a.shape[-1]) and deep.is_contiguous()
    residual=torch.empty_like(a);out=torch.empty_like(a);d=a.shape[-1]
    _add_rms[(a.numel()//d,)](a,b,deep if deep is not None else a,module.weight,residual,out,
        d,module.variance_epsilon,start,nv,deep is not None,triton.next_power_of_2(d),
        num_warps=warps,enable_fp_fusion=fusion)
    return residual,out


def add_ln(a,b,module,warps=4,fusion=False):
    assert a.shape==b.shape and a.is_contiguous() and b.is_contiguous()
    residual=torch.empty_like(a);out=torch.empty_like(a);d=a.shape[-1]
    _add_layernorm[(a.numel()//d,)](a,b,module.weight,module.bias,residual,out,d,module.eps,
        triton.next_power_of_2(d),num_warps=warps,enable_fp_fusion=fusion)
    return residual,out


@triton.jit
def _half_rows(X,W,C,S,Y,BASE,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
               STRIDE:tl.constexpr,EPS:tl.constexpr,CT:tl.constexpr,CD:tl.constexpr,
               ST:tl.constexpr,SD:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    row=BASE+tl.arange(0,R);token=row//H;head=row%H;i=tl.arange(0,B)
    ok=(row[:,None]<T*H)&(i[None,:]<D//2)
    at=token[:,None]*STRIDE+head[:,None]*D+i[None,:]
    x=tl.load(X+at,ok,0).to(tl.float32);z=tl.load(X+at+D//2,ok,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x+z*z,1)/D+EPS)
    w0=tl.load(W+i,i<D//2,0).to(tl.float32)
    w1=tl.load(W+i+D//2,i<D//2,0).to(tl.float32)
    x=((x*inv[:,None]).to(Y.dtype.element_ty).to(tl.float32)*w0[None,:]).to(Y.dtype.element_ty).to(tl.float32)
    z=((z*inv[:,None]).to(Y.dtype.element_ty).to(tl.float32)*w1[None,:]).to(Y.dtype.element_ty).to(tl.float32)
    c0=tl.load(C+token[:,None]*CT+i[None,:]*CD,ok,0).to(tl.float32)
    c1=tl.load(C+token[:,None]*CT+(i[None,:]+D//2)*CD,ok,0).to(tl.float32)
    s0=tl.load(S+token[:,None]*ST+i[None,:]*SD,ok,0).to(tl.float32)
    s1=tl.load(S+token[:,None]*ST+(i[None,:]+D//2)*SD,ok,0).to(tl.float32)
    a=(x*c0).to(Y.dtype.element_ty).to(tl.float32)
    b=(-z*s0).to(Y.dtype.element_ty).to(tl.float32)
    e=(z*c1).to(Y.dtype.element_ty).to(tl.float32)
    f=(x*s1).to(Y.dtype.element_ty).to(tl.float32)
    out=(head[:,None]*T+token[:,None])*D+i[None,:]
    tl.store(Y+out,a+b,ok);tl.store(Y+out+D//2,e+f,ok)


@triton.jit
def _half_pair(Q,K,WQ,WK,C,S,YQ,YK,T:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
               D:tl.constexpr,STRIDE:tl.constexpr,EQ:tl.constexpr,EK:tl.constexpr,
               CT:tl.constexpr,CD:tl.constexpr,ST:tl.constexpr,SD:tl.constexpr,
               R:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0);nq=tl.cdiv(T*HQ,R)
    if pid<nq:_half_rows(Q,WQ,C,S,YQ,pid*R,T,HQ,D,STRIDE,EQ,CT,CD,ST,SD,R,B)
    else:_half_rows(K,WK,C,S,YK,(pid-nq)*R,T,HK,D,STRIDE,EK,CT,CD,ST,SD,R,B)


def qk_pair(q,k,qn,kn,cos,sin,rows=4,warps=4,half=False,pad=1,fusion=False):
    assert q.shape[0]==k.shape[0]==1 and q.shape[1]==k.shape[1]
    assert q.stride(1)==k.stride(1) and q.stride(-1)==k.stride(-1)==1
    _,t,hq,d=q.shape;hk=k.shape[2]
    yq=torch.empty((1,hq,t,d),device=q.device,dtype=q.dtype)
    yk=torch.empty((1,hk,t,d),device=k.device,dtype=k.dtype)
    fn=_half_pair if half else _pair_rows
    fn[(triton.cdiv(t*hq,rows)+triton.cdiv(t*hk,rows),)](q,k,qn.weight,kn.weight,cos,sin,yq,yk,
        t,hq,hk,d,q.stride(1),qn.variance_epsilon,kn.variance_epsilon,
        cos.stride(1),cos.stride(2),sin.stride(1),sin.stride(2),rows,
        triton.next_power_of_2(d//2 if half else d)*pad,num_warps=warps,enable_fp_fusion=fusion)
    return yq,yk


@triton.jit
def _swiglu_rows(G,U,Y,I:tl.constexpr,M:tl.constexpr,STRIDE:tl.constexpr,
                 R:tl.constexpr,B:tl.constexpr,FAST:tl.constexpr):
    r=tl.program_id(0)*R+tl.arange(0,R);i=tl.program_id(1)*B+tl.arange(0,B)
    ok=(r[:,None]<M)&(i[None,:]<I)
    g=tl.load(G+r[:,None]*STRIDE+i[None,:],ok,0).to(tl.float32)
    u=tl.load(U+r[:,None]*STRIDE+i[None,:],ok,0).to(tl.float32)
    if FAST:act=g/(1.+tl.exp(-g))
    else:act=libdevice.div_rn(g,1.+libdevice.exp(-g))
    act=act.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+r[:,None]*I+i[None,:],act*u,ok)


def swiglu_rows(g,u,block=512,rows=1,warps=4,fast=False,fusion=False):
    assert g.shape==u.shape and g.stride()==u.stride() and g.stride(-1)==1
    n=g.shape[-1];m=g.numel()//n;y=torch.empty(g.shape,device=g.device,dtype=g.dtype)
    _swiglu_rows[(triton.cdiv(m,rows),triton.cdiv(n,block))](g,u,y,n,m,g.stride(-2),rows,block,fast,
        num_warps=warps,enable_fp_fusion=fusion)
    return y


@triton.jit
def _gelu(X,Y,N:tl.constexpr,B:tl.constexpr,APPROX:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);x=tl.load(X+i,i<N,0).to(tl.float32)
    if APPROX:y=.5*x*(1.+2./(1.+tl.exp(-2.*.7978845608028654*(x+.044715*x*x*x)))-1.)
    else:y=.5*x*(1.+libdevice.erf(x*.7071067811865476))
    tl.store(Y+i,y,i<N)


def gelu(x,block=512,approx=False):
    assert x.is_contiguous();out=torch.empty_like(x)
    _gelu[(triton.cdiv(x.numel(),block),)](x,out,x.numel(),block,approx,num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def _mm(A,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SA:tl.constexpr,
        BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr):
    ns=tl.program_id(0)*BN+tl.arange(0,BN);part=tl.program_id(1)
    ms=tl.arange(0,16);kk=tl.arange(0,BK)
    acc=tl.full((16,BN),0,tl.float32)
    for b in range(tl.cdiv(K,BK*SK)):
        k=(b*SK+part)*BK+kk
        a=tl.load(A+ms[:,None]*SA+k[None,:],(ms[:,None]<M)&(k[None,:]<K),0)
        w=tl.load(W+ns[None,:]*K+k[:,None],(ns[None,:]<N)&(k[:,None]<K),0)
        acc+=tl.dot(a,w)
    off=(part*M+ms[:,None])*N+ns[None,:]
    tl.store(P+off,acc,(ms[:,None]<M)&(ns[None,:]<N))


@triton.jit
def _mm_finish(P,Y,SIZE:tl.constexpr,SK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);x=tl.full((B,),0,tl.float32)
    for s in range(SK):x+=tl.load(P+s*SIZE+i,i<SIZE,0)
    tl.store(Y+i,x,i<SIZE)


def small_linear(x,w,bn=64,bk=64,split=1,warps=4):
    assert x.shape[-1]==w.shape[1] and x.stride(-1)==1 and w.is_contiguous()
    m=x.numel()//x.shape[-1];assert 1<=m<=16
    n,k=w.shape;p=torch.empty((split,m,n),device=x.device,dtype=torch.float32)
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _mm[(triton.cdiv(n,bn),split)](x,w,p,m,n,k,x.stride(-2),bn,bk,split,num_warps=warps)
    _mm_finish[(triton.cdiv(m*n,512),)](p,y,m*n,split,512,num_warps=4)
    return y


@triton.jit
def _attention(Q,K,V,Y,T:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
               QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,
               SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    head=tl.program_id(1);kv=head//(HQ//HK);mi=tl.program_id(0)*BM+tl.arange(0,BM)
    di=tl.arange(0,BD);ni=tl.arange(0,BN)
    q=tl.load(Q+head*QS+mi[:,None]*D+di[None,:],(mi[:,None]<T)&(di[None,:]<D),0)
    mx=tl.full((BM,),float('-inf'),tl.float32);den=tl.full((BM,),0,tl.float32)
    acc=tl.full((BM,BD),0,tl.float32)
    for start in range(tl.cdiv(T,BN)):
        n=start*BN+ni
        k=tl.load(K+kv*KS+n[None,:]*D+di[:,None],(n[None,:]<T)&(di[:,None]<D),0)
        scores=tl.dot(q,k)*SCALE
        scores=tl.where((mi[:,None]>=n[None,:])&(n[None,:]<T),scores,float('-inf'))
        new=tl.maximum(mx,tl.max(scores,1));alpha=tl.exp(mx-new);p=tl.exp(scores-new[:,None])
        den=den*alpha+tl.sum(p,1)
        v=tl.load(V+kv*VS+n[:,None]*D+di[None,:],(n[:,None]<T)&(di[None,:]<D),0)
        acc=acc*alpha[:,None]+tl.dot(p.to(v.dtype),v);mx=new
    result=acc/den[:,None]
    tl.store(Y+(head*T+mi[:,None])*D+di[None,:],result,(mi[:,None]<T)&(di[None,:]<D))


def attention(q,k,v,scale,bm=16,bn=32,warps=4):
    assert q.is_contiguous() and k.is_contiguous()
    # Value arrives as a view of packed QKV; materialization is included.
    v=v.contiguous();_,hq,t,d=q.shape;hk=k.shape[1]
    y=torch.empty_like(q)
    _attention[(triton.cdiv(t,bm),hq)](q,k,v,y,t,hq,hk,d,q.stride(1),k.stride(1),v.stride(1),
        scale,bm,bn,triton.next_power_of_2(d),num_warps=warps)
    return y.transpose(1,2).contiguous()
