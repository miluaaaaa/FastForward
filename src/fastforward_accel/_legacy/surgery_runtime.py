"""Private structural surgery; shared canonical checkpoint tensors never mutate."""
import functools
import hashlib
import torch
import torch.nn.functional as F
from fastforward_accel._legacy.broad_kernels import qk_pair
from fastforward_accel._legacy.limit_kernels import StaticFP8Linear, attention_direct, lm_readout
from fastforward_accel._legacy.limit_runtime import LimitJoint
from fastforward_accel._legacy.recovery_runtime import Fast as RecoveryFast
from fastforward_accel._legacy.search_profiles import keep


def digest(indices):return hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest()


def sliced(operator,rows=None,columns=None):
    # Float8 indexing is not supported by every installed torch build. FP32
    # conversion exactly represents all original FP8 values, without rescaling.
    weight=operator.weight.float()
    if rows is not None:weight=weight.index_select(0,rows)
    if columns is not None:weight=weight.index_select(1,columns)
    result=object.__new__(StaticFP8Linear)
    result.weight=weight.to(operator.weight.dtype).contiguous()
    result.weight_scale=operator.weight_scale;result.input_scale=operator.input_scale
    result.inverse_scale=operator.inverse_scale
    return result


class StructuralJoint(LimitJoint):
    def __init__(self,*args,shared,**kwargs):
        super().__init__(*args,**kwargs);self.shared=shared

    def configure(self,spec,prototype):
        original=self.features['static_fp8'];assert original=='all'
        if 'attention' in self.shared:self.features['static_fp8']=False
        try:super().configure(spec,prototype)
        finally:self.features['static_fp8']=original
        lm=self.m.vl_model.language_model;s=self.features.get('structural',{})
        if 'attention' not in self.shared:
            assert self.language_indices==list(range(36))
            self.shared['attention']=dict(self.static_attention);self.shared['mlp']=dict(self.static_mlp)
        else:
            self.static_attention={id(lm.layers[j].self_attn):self.shared['attention'][id(lm.layers[j].self_attn)] for j in self.language_indices}
            self.static_mlp={id(lm.layers[j].mlp):self.shared['mlp'][id(lm.layers[j].mlp)] for j in self.language_indices}
        self.spec['numeric_format']='FP8 E4M3FN; slices retain incumbent scales; folded matrices requantized at setup'
        self.surgery=s;self.head_shapes={};self.fold_gain={};changes=[]
        if 'vision_count' in s:
            v=self.vg.v;protected=sorted(set(v.deepstack_visual_indexes)|{len(v.blocks)-1})
            chosen=keep(len(v.blocks),s['vision_count'],'uniform',protected)
            self.vision_indices=chosen;self.vg.active_indices=set(chosen)
            self.spec['vision_indices']=chosen
        eligible=[j for j in self.language_indices if j not in [0,1,2,35]]
        assert s.get('skip_ffn',0)<=len(eligible) and s.get('skip_attention',0)<=len(eligible)
        self.omit_ffn=set(eligible[-s['skip_ffn']:]) if s.get('skip_ffn') else set()
        self.omit_attention=set(eligible[-s['skip_attention']:]) if s.get('skip_attention') else set()
        for j,k in s.get('pairs',[]):
            assert j in self.language_indices and k not in self.language_indices
            a=lm.layers[j];b=lm.layers[k]
            def merged(x,y):return StaticFP8Linear(((x.float()+y.float())*.5).to(x.dtype),self.features['fp8_inverse_scale'])
            self.static_attention[id(a.self_attn)]=(merged(self.packed[j][0],self.packed[k][0]),
                merged(a.self_attn.o_proj.weight,b.self_attn.o_proj.weight))
            self.static_mlp[id(a.mlp)]=(merged(self.packed[j][1],self.packed[k][1]),
                merged(a.mlp.down_proj.weight,b.mlp.down_proj.weight))
            self.fold_gain[j]=s['fold_gain']
        ratio=s.get('ffn_ratio',1.)
        if ratio<1:
            selected=self.language_indices[-18:] if s.get('ffn_scope')=='last18' else self.language_indices
            for j in selected:
                if j in self.omit_ffn:continue
                layer=lm.layers[j];gate,down=self.static_mlp[id(layer.mlp)]
                width=gate.weight.shape[0]//2;n=max(128,int(width*ratio)//128*128)
                key=('ffn_order',j)
                if key not in self.shared:
                    # Input/label independent importance, calculated only once.
                    self.shared[key]=layer.mlp.down_proj.weight.float().square().sum(0).argsort(descending=True)
                order=self.shared[key][:n].sort().values
                rows=torch.cat((order,order+width))
                self.static_mlp[id(layer.mlp)]=(sliced(gate,rows=rows),sliced(down,columns=order))
                changes.append(dict(kind='ffn_width',layer=j,original=width,retained=n,index_sha256=digest(order)))
        if s.get('heads'):
            selected=self.language_indices[-18:] if s.get('head_scope')=='last18' else self.language_indices
            for j in selected:
                if j in self.omit_attention:continue
                a=lm.layers[j].self_attn;d=a.head_dim;nq=a.q_proj.out_features//d;nk=a.k_proj.out_features//d
                count=s['heads'];assert nq==32 and nk==8
                if count<=nk:
                    assert nk%count==0
                    keys=list(range(0,nk,nk//count));queries=[k*(nq//nk) for k in keys]
                else:
                    assert count%nk==0 and count<=nq
                    keys=list(range(nk));queries=[k*(nq//nk)+r for k in keys for r in range(count//nk)]
                qr=torch.tensor([h*d+i for h in queries for i in range(d)],device='cuda')
                kr=torch.tensor([h*d+i for h in keys for i in range(d)],device='cuda')
                rows=torch.cat((qr,kr+nq*d,kr+(nq+nk)*d))
                qkv,out=self.static_attention[id(a)]
                self.static_attention[id(a)]=(sliced(qkv,rows=rows),sliced(out,columns=qr))
                self.head_shapes[id(a)]=(len(queries),len(keys))
                changes.append(dict(kind='attention_width',layer=j,query_heads=queries,kv_heads=keys))
        linear_elements=0;layouts=[]
        for j in self.language_indices:
            layer=lm.layers[j]
            if j not in self.omit_attention:linear_elements+=sum(op.weight.numel() for op in self.static_attention[id(layer.self_attn)])
            if j not in self.omit_ffn:linear_elements+=sum(op.weight.numel() for op in self.static_mlp[id(layer.mlp)])
            layouts.append(dict(layer=j,attention=[] if j in self.omit_attention else
                [list(op.weight.shape) for op in self.static_attention[id(layer.self_attn)]],
                ffn=[] if j in self.omit_ffn else [list(op.weight.shape) for op in self.static_mlp[id(layer.mlp)]]))
        self.spec['architecture_search']=dict(language_indices=self.language_indices,
            vision_indices=self.vision_indices,skipped_attention=sorted(self.omit_attention),
            skipped_ffn=sorted(self.omit_ffn),fold_pairs=s.get('pairs',[]),fold_gain=s.get('fold_gain'),
            matrix_changes=changes,linear_layouts=layouts,language_linear_weight_elements=linear_elements,
            frozen=True,training=False,readout='Original option-token LM rows',
            folded_norms='First retained block norms; mean attention/MLP matrices; explicit residual gain')

    def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
        if id(a) not in self.head_shapes:return super().attention(a,w,hidden_states,position_embeddings,attention_mask,**kwargs)
        nq,nk=self.head_shapes[id(a)];d=a.head_dim;shape=hidden_states.shape[:-1]
        projection,output=self.static_attention[id(a)]
        packed=projection(hidden_states).view(*shape,-1)
        q,k,v=packed.split([nq*d,nk*d,nk*d],-1)
        q,k=qk_pair(q.view(*shape,nq,d),k.view(*shape,nk,d),a.q_norm,a.k_norm,*position_embeddings,4,4,False)
        v=v.view(*shape,nk,d).transpose(1,2)
        out=attention_direct(q,k,v,a.scaling,self.features.get('attn_bm',32),self.features.get('attn_bn',32),4)
        return output(out.reshape(*shape,-1)).view(*shape,-1),None

    def core_joint(self,args,native=False):
        if not self.surgery:return super().core_joint(args,native)
        assert not native
        rgb,ids,picks,mask=args;lm=self.m.vl_model.language_model
        visual,deep=self.vision(rgb);assert len(deep)==3
        h=self.m.vl_model.get_input_embeddings()(ids);h[0].index_copy_(0,self.vi,visual)
        rope=lm.rotary_emb(h,self.pos[ids.shape[1]])
        normalized=self.norm(h,lm.layers[self.language_indices[0]].input_layernorm)
        chosen=None
        for offset,j in enumerate(self.language_indices):
            layer=lm.layers[j];qw,gw=self.packed[j];final=offset==len(self.language_indices)-1
            if j in self.omit_attention:
                residual=h;post=self.norm(h,layer.post_attention_layernorm)
            else:
                attn,_=self.attention(layer.self_attn,qw,hidden_states=normalized,position_embeddings=rope,attention_mask=None)
                residual,post=self.post_attention(h,attn,layer)
            gain=self.fold_gain.get(j,1.)
            if gain!=1.:residual=h+(residual-h)*gain
            if final:
                chosen=residual[0].index_select(0,picks[8:9])
                if j not in self.omit_ffn:
                    value=self.mlp(layer.mlp,gw,post[0].index_select(0,picks[8:9]))
                    if gain!=1.:value=value*gain
                    chosen=chosen+value
            else:
                injection=deep[j] if j<len(deep) else None
                next_norm=lm.layers[self.language_indices[offset+1]].input_layernorm
                if j in self.omit_ffn:
                    h=self.inject_reference(residual,injection);normalized=self.norm(h,next_norm)
                else:
                    value=self.mlp(layer.mlp,gw,post)
                    if gain!=1.:value=value*gain
                    h,normalized=self.carry(residual,value,injection,next_norm,self.image_start,self.Nv)
        return lm_readout(chosen.contiguous(),lm.norm,self.option_weight)


class Fast(RecoveryFast):
    def __init__(self,reference):
        super().__init__(reference);self.shared={}
        self.runtime_class=functools.partial(StructuralJoint,shared=self.shared)
        self.private_versions=None

    def assert_frozen(self):
        super().assert_frozen()
        versions=tuple(op.weight._version for kind in ['attention','mlp'] for ops in self.shared[kind].values() for op in ops)
        if self.private_versions is None:self.private_versions=versions
        assert self.private_versions==versions,'Incumbent private FP8 weights mutated'
