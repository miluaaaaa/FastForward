"""Private exact residual-add/square fusion; PyTorch reduction stays unchanged."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from shunong_accel._legacy.serving_dynamic_norm import _finish, fused_norm
from shunong_accel._legacy.sub10_runtime import Joint


@triton.jit
def _add_square(A, B, R, S, N, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A+i, i<N, other=0).to(tl.float32)
    b = tl.load(B+i, i<N, other=0).to(tl.float32)
    # Match the materialized BF16 torch.add before squaring in FP32.
    r = (a+b).to(R.dtype.element_ty).to(tl.float32)
    tl.store(R+i, r, i<N)
    tl.store(S+i, r*r, i<N)


def residual_norm(a, b, module):
    assert a.is_contiguous() and b.is_contiguous() and a.shape == b.shape
    assert a.dtype == b.dtype == module.weight.dtype == torch.bfloat16
    residual = torch.empty_like(a)
    square = torch.empty_like(a, dtype=torch.float32)
    n = a.numel()
    _add_square[(triton.cdiv(n,1024),)](a,b,residual,square,n,1024,enable_fp_fusion=False)
    variance = square.mean(-1,keepdim=True)
    out = torch.empty_like(a)
    _finish[(triton.cdiv(n,1024),)](residual,variance,module.weight,out,n,a.shape[-1],
                                  module.variance_epsilon,1024,enable_fp_fusion=False)
    return residual, out


class ResidualJoint(Joint):
    validate_layers = False

    def core_joint(self,args,native=False):
        if native:
            return super().core_joint(args,True)
        rgb,ids,picks,mask=args
        b=ids.shape[1]
        lm=self.m.vl_model.language_model
        visual,deep=self.vision(rgb)
        h=self.m.vl_model.get_input_embeddings()(ids)
        h[0].index_copy_(0,self.vi,visual)
        pos=self.pos[b]
        rope=lm.rotary_emb(h,pos)
        cache=torch.arange(b,device='cuda')
        for j in self.language_indices:
            layer=lm.layers[j]
            qw,gw=self.packed[j]
            attn,_=self.attention(layer.self_attn,qw,hidden_states=self.norm(h,layer.input_layernorm),
                position_embeddings=rope,attention_mask=None,position_ids=pos[0],cache_position=cache,use_cache=False)
            residual,normalized=residual_norm(h,attn,layer.post_attention_layernorm)
            if self.validate_layers:
                reference=h+attn
                assert torch.equal(residual,reference), ('residual',j)
                assert torch.equal(normalized,fused_norm(reference,layer.post_attention_layernorm)), ('norm',j)
                self.layer_checks.append(dict(layer=j,shape=list(h.shape),residual_exact=True,norm_exact=True))
            h=residual+self.mlp(layer.mlp,gw,normalized)
            if j<len(deep):
                add=torch.zeros_like(h)
                add[0].index_copy_(0,self.vi,deep[j])
                h=h+add
        chosen=h[0].index_select(0,picks)
        if self.readout=='head':
            return self.heads[str(self.depth)].score(chosen[8][None],chosen[:8][None],mask)
        return F.linear(lm.norm(chosen[8][None]).to(self.m.dtype),self.option_weight).float()
