"""Private attention layouts and small terminal readout, without library patches."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _attention_direct(Q,K,V,Y,POSITIONS,S:tl.constexpr,T:tl.constexpr,HQ:tl.constexpr,
                      HK:tl.constexpr,D:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr,
                      VH:tl.constexpr,VT:tl.constexpr,SCALE:tl.constexpr,
                      BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr,
                      SELECTED:tl.constexpr,TRIANGULAR:tl.constexpr):
    head=tl.program_id(1);kv=head//(HQ//HK)
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    di=tl.arange(0,BD);ni=tl.arange(0,BN)
    q=tl.load(Q+head*QS+mi[:,None]*D+di[None,:],(mi[:,None]<S)&(di[None,:]<D),0)
    if SELECTED:position=tl.load(POSITIONS+mi,mi<S,0)
    else:position=mi
    mx=tl.full((BM,),float('-inf'),tl.float32)
    den=tl.full((BM,),0,tl.float32);acc=tl.full((BM,BD),0,tl.float32)
    if TRIANGULAR and not SELECTED:blocks=tl.cdiv(tl.minimum(T,(tl.program_id(0)+1)*BM),BN)
    else:blocks=tl.cdiv(T,BN)
    for start in range(blocks):
        n=start*BN+ni
        key=tl.load(K+kv*KS+n[None,:]*D+di[:,None],(n[None,:]<T)&(di[:,None]<D),0)
        score=tl.dot(q,key)*SCALE
        score=tl.where((position[:,None]>=n[None,:])&(n[None,:]<T),score,float('-inf'))
        new=tl.maximum(mx,tl.max(score,1));alpha=tl.exp(mx-new);p=tl.exp(score-new[:,None])
        den=den*alpha+tl.sum(p,1)
        value=tl.load(V+kv*VH+n[:,None]*VT+di[None,:],(n[:,None]<T)&(di[None,:]<D),0)
        acc=acc*alpha[:,None]+tl.dot(p.to(value.dtype),value);mx=new
    tl.store(Y+(mi[:,None]*HQ+head)*D+di[None,:],acc/den[:,None],(mi[:,None]<S)&(di[None,:]<D))


def attention_direct(q,k,v,scale,bm=16,bn=64,warps=4,positions=None,triangular=True):
    assert q.is_contiguous() and k.is_contiguous() and q.shape[0]==k.shape[0]==v.shape[0]==1
    _,hq,s,d=q.shape;_,hk,t,kd=k.shape
    assert kd==d and v.shape==(1,hk,t,d) and v.stride(-1)==1 and hq%hk==0
    if positions is not None:assert positions.shape==(s,) and positions.is_contiguous()
    y=torch.empty((1,s,hq,d),device=q.device,dtype=q.dtype)
    _attention_direct[(triton.cdiv(s,bm),hq)](q,k,v,y,q if positions is None else positions,
        s,t,hq,hk,d,q.stride(1),k.stride(1),v.stride(1),v.stride(2),scale,bm,bn,
        triton.next_power_of_2(d),positions is not None,triangular,num_warps=warps)
    return y


@triton.jit
def _lm_readout(X,NORM,W,Y,D:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    option=tl.program_id(0);i=tl.arange(0,B)
    x=tl.load(X+i,i<D,0).to(tl.float32)
    weight=tl.load(NORM+i,i<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    x=((x*inv).to(NORM.dtype.element_ty).to(tl.float32)*weight).to(NORM.dtype.element_ty).to(tl.float32)
    w=tl.load(W+option*D+i,i<D,0).to(tl.float32)
    out=tl.sum(x*w,0).to(NORM.dtype.element_ty).to(tl.float32)
    tl.store(Y+option,out)


def lm_readout(x,norm,weight):
    assert x.numel()==x.shape[-1]==weight.shape[1] and x.is_contiguous() and weight.is_contiguous()
    assert x.dtype==weight.dtype==norm.weight.dtype==torch.bfloat16
    y=torch.empty((1,weight.shape[0]),device=x.device,dtype=torch.float32)
    _lm_readout[(weight.shape[0],)](x,norm.weight,weight,y,x.shape[-1],norm.variance_epsilon,
        triton.next_power_of_2(x.shape[-1]),num_warps=4,enable_fp_fusion=False)
    return y


@triton.jit
def _readout_rms(X,Y,D:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,B)
    x=tl.load(X+row*D+i,i<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+1e-6)
    tl.store(Y+row*D+i,x*inv,i<D)


def head_readout(chosen,head,mask):
    assert chosen.ndim==2 and chosen.shape[0]==9 and chosen.is_contiguous()
    normalized=torch.empty(chosen.shape,device=chosen.device,dtype=torch.float32)
    _readout_rms[(9,)](chosen,normalized,chosen.shape[-1],triton.next_power_of_2(chosen.shape[-1]),num_warps=4)
    return head.head.score(normalized[8][None],normalized[:8][None],mask)


@triton.jit
def _gateup_swiglu(X,W,Y,M:tl.constexpr,I:tl.constexpr,K:tl.constexpr,XS:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    gate=tl.full((BM,BN),0,tl.float32);up=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        ks=block*BK+kk
        x=tl.load(X+rows[:,None]*XS+ks[None,:],(rows[:,None]<M)&(ks[None,:]<K),0)
        g=tl.load(W+cols[None,:]*K+ks[:,None],(cols[None,:]<I)&(ks[:,None]<K),0)
        u=tl.load(W+(cols[None,:]+I)*K+ks[:,None],(cols[None,:]<I)&(ks[:,None]<K),0)
        gate+=tl.dot(x,g);up+=tl.dot(x,u)
    gate=gate.to(Y.dtype.element_ty).to(tl.float32);up=up.to(Y.dtype.element_ty).to(tl.float32)
    act=libdevice.div_rn(gate,1.+libdevice.exp(-gate)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+rows[:,None]*I+cols[None,:],act*up,(rows[:,None]<M)&(cols[None,:]<I))


def gateup_swiglu(x,w,bm=32,bn=64,bk=64,warps=4):
    assert w.is_contiguous() and x.stride(-1)==1 and w.shape[0]%2==0 and x.shape[-1]==w.shape[1]
    i=w.shape[0]//2;m=x.numel()//x.shape[-1]
    y=torch.empty((*x.shape[:-1],i),device=x.device,dtype=x.dtype)
    _gateup_swiglu[(triton.cdiv(m,bm),triton.cdiv(i,bn))](x,w,y,m,i,x.shape[-1],x.stride(-2),bm,bn,bk,
        num_warps=warps,num_stages=3,enable_fp_fusion=False)
    return y


@triton.jit
def _static_fp8(X,Q,M:tl.constexpr,K:tl.constexpr,INV:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.load(X+i,i<M*K,0).to(tl.float32)*INV
    tl.store(Q+i,tl.minimum(448.,tl.maximum(-448.,value)),i<tl.cdiv(M,16)*16*K)


class StaticFP8Linear:
    """Setup-only tensor-scale weights; fixed activation scale and fast accumulation.

    Saturation is an explicit approximation, evaluated by actual model scores.
    """
    def __init__(self,weight,inverse_scale=16.):
        w=weight.detach().float();scale=w.abs().amax().clamp_min(1e-12)/448.
        self.weight=(w/scale).clamp(-448,448).to(torch.float8_e4m3fn).contiguous()
        self.weight_scale=scale.reshape(1)
        self.inverse_scale=inverse_scale
        self.input_scale=torch.tensor([1./inverse_scale],device=w.device,dtype=torch.float32)

    def __call__(self,x):
        x=x.reshape(-1,x.shape[-1]).contiguous();m,k=x.shape
        q=torch.empty((triton.cdiv(m,16)*16,k),device=x.device,dtype=torch.float8_e4m3fn)
        _static_fp8[(triton.cdiv(q.numel(),1024),)](x,q,m,k,self.inverse_scale,1024,num_warps=4)
        y=torch._scaled_mm(q,self.weight.T,scale_a=self.input_scale,scale_b=self.weight_scale,
                           out_dtype=torch.bfloat16,use_fast_accum=True)
        return y[:m]
