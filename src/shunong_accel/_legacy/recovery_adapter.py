"""Local residual repair with exact serving forward and a declared STE backward.

Canonical model tensors are frozen. The backward is a differentiable reference
of the final block; FP8 casts use straight-through derivatives. It is not the
mathematical derivative of the discontinuous quantized serving function.
"""
import functools
import torch
from torch import nn
from torch.nn import functional as F
from shunong_accel._legacy.surgery_runtime import StructuralJoint, Fast as StructuralFast
from shunong_accel._legacy.limit_kernels import lm_readout


def rms(x, module=None):
    y=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+(module.variance_epsilon if module is not None else 1e-6))).to(x.dtype)
    return y if module is None else y*module.weight.to(x.dtype)


class Repair(nn.Module):
    def __init__(self, width, rank=32, seed=71):
        super().__init__()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            self.down=nn.Parameter(torch.randn(rank,width,device='cuda')/width**.5)
            self.up=nn.Parameter(torch.zeros(width,rank,device='cuda'))
        self.bias=nn.Parameter(torch.zeros(width,device='cuda'))

    def forward(self,h):
        x=rms(h)
        z=F.linear(x,self.down.to(h.dtype))
        return h+(F.linear(z,self.up.to(h.dtype))+self.bias.to(h.dtype))


class FP8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,operator):
        ctx.operator=operator;ctx.shape=x.shape
        return operator(x).reshape(*x.shape[:-1],operator.weight.shape[0])

    @staticmethod
    def backward(ctx,gradient):
        op=ctx.operator
        weight=(op.weight.float()*op.weight_scale).to(gradient.dtype)
        return F.linear(gradient,weight.T).reshape(ctx.shape),None


def surrogate_tail(rt,h,picks,rope):
    """Backward-only QAT reference: same frozen FP8 linear forwards, torch SDPA."""
    lm=rt.m.vl_model.language_model;layer=lm.layers[35];a=layer.self_attn
    projection,output=rt.static_attention[id(a)];gate,down=rt.static_mlp[id(layer.mlp)]
    z=FP8STE.apply(rms(h,layer.input_layernorm),projection)
    q,k,v=z.split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],-1)
    def rotary(x,module):
        x=rms(x.view(*h.shape[:-1],-1,a.head_dim),module).transpose(1,2)
        c,s=rope;c=c[:,None];s=s[:,None]
        left,right=x.chunk(2,-1)
        return (x*c).to(x.dtype)+(torch.cat((-right,left),-1)*s).to(x.dtype)
    q=rotary(q,a.q_norm);k=rotary(k,a.k_norm)
    v=v.view(*h.shape[:-1],-1,a.head_dim).transpose(1,2)
    groups=q.shape[1]//k.shape[1]
    # Math SDPA has a backward on the installed build, including second calls.
    out=F.scaled_dot_product_attention(q,k.repeat_interleave(groups,1),v.repeat_interleave(groups,1),is_causal=True,scale=a.scaling)
    residual=h+FP8STE.apply(out.transpose(1,2).reshape(*h.shape[:-1],-1),output)
    selected=residual[0].index_select(0,picks[8:9])
    post=rms(residual,layer.post_attention_layernorm)[0].index_select(0,picks[8:9])
    g,u=FP8STE.apply(post,gate).chunk(2,-1)
    chosen=selected+FP8STE.apply(F.silu(g)*u,down)
    return F.linear(rms(chosen,lm.norm),rt.option_weight).float()


class ExactTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,picks,cos,sin,rt):
        ctx.save_for_backward(h,picks,cos,sin);ctx.rt=rt
        return rt.final(h,picks,(cos,sin))

    @staticmethod
    def backward(ctx,gradient):
        h,picks,cos,sin=ctx.saved_tensors
        with torch.enable_grad():
            x=h.detach().requires_grad_(True)
            z=surrogate_tail(ctx.rt,x,picks,(cos,sin))
            dh,=torch.autograd.grad(z,x,gradient)
        return dh,None,None,None,None


class RepairJoint(StructuralJoint):
    def __init__(self,*args,bank,**kwargs):
        super().__init__(*args,**kwargs);self.bank=bank;self.capture=False;self.captured=None

    def configure(self,spec,prototype):
        super().configure(spec,prototype)
        self.repair=self.bank.get(self.features.get('repair_id'))
        if self.repair is not None:
            self.spec['task_repair']=dict(rank=32,placement='before original final block35',
                trainable_parameters=sum(p.numel() for p in self.repair.parameters()),
                training_forward='exact deployed FP8 final block',backward='declared FP8 straight-through reference')

    def final(self,h,picks,rope):
        layer=self.m.vl_model.language_model.layers[35];qw,gw=self.packed[35]
        normalized=self.norm(h,layer.input_layernorm)
        attn,_=self.attention(layer.self_attn,qw,hidden_states=normalized,position_embeddings=rope,attention_mask=None)
        residual,post=self.post_attention(h,attn,layer)
        chosen=residual[0].index_select(0,picks[8:9])+self.mlp(layer.mlp,gw,post[0].index_select(0,picks[8:9]))
        return lm_readout(chosen.contiguous(),self.m.vl_model.language_model.norm,self.option_weight)

    def core_joint(self,args,native=False):
        assert not native
        rgb,ids,picks,mask=args;lm=self.m.vl_model.language_model
        visual,deep=self.vision(rgb);h=lm.embed_tokens(ids);h[0].index_copy_(0,self.vi,visual)
        rope=lm.rotary_emb(h,self.pos[ids.shape[1]])
        normalized=self.norm(h,lm.layers[self.language_indices[0]].input_layernorm)
        for offset,j in enumerate(self.language_indices):
            if j==35:
                if self.capture:self.captured=(h.detach().clone(),picks.clone(),tuple(x.clone() for x in rope))
                if self.repair is not None:h=self.repair(h)
                return self.final(h,picks,rope)
            layer=lm.layers[j];qw,gw=self.packed[j]
            attn,_=self.attention(layer.self_attn,qw,hidden_states=normalized,position_embeddings=rope,attention_mask=None)
            residual,post=self.post_attention(h,attn,layer)
            h,normalized=self.carry(residual,self.mlp(layer.mlp,gw,post),deep[j] if j<3 else None,
                lm.layers[self.language_indices[offset+1]].input_layernorm,self.image_start,self.Nv)
        raise AssertionError('Original final layer must be retained')


class Fast(StructuralFast):
    def __init__(self,reference):
        super().__init__(reference);self.bank={}
        self.runtime_class=functools.partial(RepairJoint,shared=self.shared,bank=self.bank)


def task_loss(logits,teacher,gold,objective,swapped=None,teacher_swapped=None):
    # Slice to real candidates before this boundary. No padded answer classes.
    kl=F.kl_div((logits/2).log_softmax(-1),(teacher/2).softmax(-1),reduction='batchmean')*4
    ce=F.cross_entropy(logits,torch.tensor([gold],device=logits.device))
    value=kl
    if objective=='joint':
        assert swapped is not None and teacher_swapped is not None
        sk=F.kl_div((swapped/2).log_softmax(-1),(teacher_swapped/2).softmax(-1),reduction='batchmean')*4
        value=value+.5*ce+.5*sk
    else:assert objective=='distill'
    return value
