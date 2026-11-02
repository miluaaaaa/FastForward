"""Weight-preserving Transformer block pipeline with inter-layer residual fusion."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from shunong_accel._legacy.serving_dynamic_norm import _finish, fused_norm
from shunong_accel._legacy.residual_runtime import residual_norm
from shunong_accel._legacy.sub10_runtime import Joint


@triton.jit
def _carry_square(A,B,D,R,S,N,H:tl.constexpr,START:tl.constexpr,NV:tl.constexpr,
                  HAS_DEEP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N
    a=tl.load(A+i,valid,other=0).to(tl.float32)
    b=tl.load(B+i,valid,other=0).to(tl.float32)
    r=(a+b).to(R.dtype.element_ty).to(tl.float32)
    if HAS_DEEP:
        token=i//H
        d=tl.load(D+(token-START)*H+i%H,valid & (token>=START) & (token<START+NV),other=0).to(tl.float32)
        # The historical dense zero-injection adds +0 outside image tokens too.
        r=(r+d).to(R.dtype.element_ty).to(tl.float32)
    tl.store(R+i,r,valid)
    tl.store(S+i,r*r,valid)


def carry_norm(a,b,deep,module,start,nv):
    assert a.is_contiguous() and b.is_contiguous() and a.shape==b.shape
    assert a.dtype==b.dtype==module.weight.dtype==torch.bfloat16
    if deep is not None:
        assert deep.is_contiguous() and deep.dtype==a.dtype
        assert deep.shape==(nv,a.shape[-1])
    residual=torch.empty_like(a)
    square=torch.empty_like(a,dtype=torch.float32)
    n=a.numel()
    _carry_square[(triton.cdiv(n,1024),)](a,b,deep if deep is not None else a,residual,square,
        n,a.shape[-1],start,nv,deep is not None,1024,enable_fp_fusion=False)
    variance=square.mean(-1,keepdim=True)
    out=torch.empty_like(a)
    _finish[(triton.cdiv(n,1024),)](residual,variance,module.weight,out,n,a.shape[-1],
        module.variance_epsilon,1024,enable_fp_fusion=False)
    return residual,out


class BlockJoint(Joint):
    validate_layers=False
    carry=staticmethod(carry_norm)

    def configure(self,spec,prototype):
        super().configure(spec,prototype)
        assert spec.get('norm')=='exact'
        self.image_start=int(self.vi[0].item())
        self.selective_tail=False

    def inject_reference(self,h,deep):
        if deep is not None:
            add=torch.zeros_like(h)
            add[0].index_copy_(0,self.vi,deep)
            h=h+add
        return h

    def core_joint(self,args,native=False):
        if native:
            return super().core_joint(args,True)
        rgb,ids,picks,mask=args
        lm=self.m.vl_model.language_model
        visual,deep=self.vision(rgb)
        h=self.m.vl_model.get_input_embeddings()(ids)
        h[0].index_copy_(0,self.vi,visual)
        pos=self.pos[ids.shape[1]]
        rope=lm.rotary_emb(h,pos)
        cache=torch.arange(ids.shape[1],device='cuda')
        normalized=self.norm(h,lm.layers[self.language_indices[0]].input_layernorm)
        chosen=None
        for offset,j in enumerate(self.language_indices):
            layer=lm.layers[j]
            qw,gw=self.packed[j]
            attn,_=self.attention(layer.self_attn,qw,hidden_states=normalized,
                position_embeddings=rope,attention_mask=None,position_ids=pos[0],cache_position=cache,use_cache=False)
            residual,post=self.post_attention(h,attn,layer)
            check=dict(layer=j,shape=list(h.shape),residual_exact=True,norm_exact=True)
            final=offset==len(self.language_indices)-1
            injection=deep[j] if j<len(deep) else None
            if final and self.selective_tail:
                # Attention retains the entire context. Only token-independent
                # final FFN work that cannot influence this readout is omitted.
                selected=picks if self.readout=='head' else picks[8:9]
                narrow_residual=residual[0].index_select(0,selected)
                narrow_post=post[0].index_select(0,selected)
                chosen=narrow_residual+self.mlp(layer.mlp,gw,narrow_post)
                if self.validate_layers:
                    full=residual+self.mlp(layer.mlp,gw,post)
                    assert torch.equal(chosen,full[0].index_select(0,selected)),('selected_tail',j)
                    check['selected_tail_exact']=True
                    check['selected_tokens']=int(selected.numel())
                if injection is not None:
                    raise RuntimeError('Selective terminal FFN requires no terminal deepstack injection')
            else:
                mlp=self.mlp(layer.mlp,gw,post)
                if final:
                    h=self.inject_reference(residual+mlp,injection)
                else:
                    next_layer=lm.layers[self.language_indices[offset+1]]
                    h,normalized=self.carry(residual,mlp,injection,next_layer.input_layernorm,self.image_start,self.Nv)
                    if self.validate_layers:
                        reference=self.inject_reference(residual+mlp,injection)
                        assert torch.equal(h,reference),('carry_residual',j)
                        assert torch.equal(normalized,self.norm(reference,next_layer.input_layernorm)),('carry_norm',j)
                        check.update(carry_residual_exact=True,carry_norm_exact=True,deepstack=injection is not None)
            if self.validate_layers:
                self.layer_checks.append(check)
        if chosen is None:
            chosen=h[0].index_select(0,picks)
        if self.readout=='head':
            return self.heads[str(self.depth)].score(chosen[8][None],chosen[:8][None],mask)
        last=chosen if self.selective_tail else chosen[8][None]
        return F.linear(lm.norm(last).to(self.m.dtype),self.option_weight).float()

    def post_attention(self,h,attn,layer):
        residual,normalized=residual_norm(h,attn,layer.post_attention_layernorm)
        if self.validate_layers:
            reference=h+attn
            assert torch.equal(residual,reference),('attention_residual',)
            assert torch.equal(normalized,fused_norm(reference,layer.post_attention_layernorm)),('attention_norm',)
        return residual,normalized
