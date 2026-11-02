"""Bounded multi-family search built on the latest resident LowerJoint."""
import contextlib
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS
from fastforward_accel._legacy.lower_runtime import LowerJoint, LowerVision
from fastforward_accel._legacy.lower_kernels import norm_rope
from fastforward_accel._legacy.search_kernels import swiglu, _vision_rope
from fastforward_accel._legacy.rotary_exact_runtime import norm_rotary
import fastforward_accel._legacy.broad_kernels as kernels


def backend(name):
    return sdpa_kernel(getattr(SDPBackend,name)) if name else contextlib.nullcontext()


class BroadVision(LowerVision):
    def core(self,pixels,geometry):
        f=self.owner.features;v=self.v;pos,cos,sin=geometry
        if not any(k in f for k in ('patch_mode','ln_warps','ln_fusion','vision_rope_warps',
                'vision_gelu','vision_attention_backend','vision_depth_mode')):
            return super().core(pixels,geometry)
        pe=v.patch_embed;mode=f.get('patch_mode')
        if mode=='linear':h=F.linear(pixels.to(pe.proj.weight.dtype),pe.proj.weight.flatten(1),pe.proj.bias)+pos
        elif mode=='channels_last':
            z=pixels.view(-1,pe.in_channels,pe.temporal_patch_size,pe.patch_size,pe.patch_size)
            h=pe.proj(z.to(pe.proj.weight.dtype).contiguous(memory_format=torch.channels_last_3d)).view(-1,pe.embed_dim)+pos
        else:h=pe(pixels)+pos
        h=h.reshape(h.shape[0],-1);n=h.shape[0];deep=[]
        indices=sorted(self.active_indices)
        if not indices:return v.merger(h),deep
        x=v.blocks[indices[0]].norm1(h)
        for offset,j in enumerate(indices):
            block=v.blocks[j];a=block.attn
            q,k,value=a.qkv(x).reshape(n,3,a.num_heads,-1).permute(1,0,2,3).unbind(0)
            yq=torch.empty_like(q);yk=torch.empty_like(k);d=q.shape[-1];b=f.get('vision_block',512)
            _vision_rope[((q.numel()+b-1)//b,)](q,k,cos,sin,yq,yk,q.numel(),a.num_heads,d,q.stride(0),b,
                num_warps=f.get('vision_rope_warps',4),enable_fp_fusion=False)
            q=yq.transpose(0,1).unsqueeze(0);k=yk.transpose(0,1).unsqueeze(0);value=value.transpose(0,1).unsqueeze(0)
            with backend(f.get('vision_attention_backend')):
                attn=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,value,attention_mask=None,scaling=a.scaling,dropout=0.,is_causal=False)[0]
            projected=a.proj(attn.reshape(n,-1).contiguous())
            h,post=kernels.add_ln(h,projected,block.norm2,f.get('ln_warps',4),f.get('ln_fusion',False))
            if f.get('vision_gelu'):
                mlp=block.mlp
                z=kernels.gelu(mlp.linear_fc1(post),f.get('gelu_block',512),f['vision_gelu']=='fast')
                z=mlp.linear_fc2(z)
            else:z=block.mlp(post)
            if offset+1<len(indices):
                h,x=kernels.add_ln(h,z,v.blocks[indices[offset+1]].norm1,f.get('ln_warps',4),f.get('ln_fusion',False))
            else:h=h+z
            if j in v.deepstack_visual_indexes:deep.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(j)](h))
        return v.merger(h),deep


class BroadJoint(LowerJoint):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.vg=BroadVision(self.m);self.vg.owner=self

    def configure(self,spec,prototype):
        super().configure(spec,prototype)
        f=self.features
        if f.get('vision_depth_mode'):
            count=f['vision_count'];original=spec['vision_indices']
            selected=original[:count] if f['vision_depth_mode']=='prefix' else original[-count:]
            self.vision_indices=selected;self.vg.active_indices=set(selected);self.spec['vision_indices']=selected
        if f.get('ffn_selection'):
            assert 0<f['ffn_ratio']<1 and not self.fp8_mlp
            import hashlib
            selected={}
            for j in self.language_indices:
                module=self.m.vl_model.language_model.layers[j].mlp
                gate,up=self.packed[j][1].chunk(2,0);down=module.down_proj.weight
                importance=down.float().square().sum(0).sqrt()
                if f['ffn_selection']=='gate_up_down':
                    importance=importance*gate.float().square().sum(1).sqrt()*up.float().square().sum(1).sqrt()
                else:assert f['ffn_selection']=='down'
                n=max(128,int(gate.shape[0]*f['ffn_ratio'])//128*128)
                indices=importance.topk(n,sorted=False).indices.sort().values
                packed=torch.cat((gate.index_select(0,indices),up.index_select(0,indices))).contiguous()
                output=down.index_select(1,indices).contiguous()*f.get('ffn_rescale',1.)
                self.narrow[id(module)]=(packed,output)
                selected[str(j)]=dict(count=n,sha256=hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest())
            self.spec['ffn_channel_selection']=dict(method=f['ffn_selection'],rescale=f.get('ffn_rescale',1.),layers=selected)
        self.spec['broad_operators']={k:v for k,v in f.items() if k.startswith(('qk_','rms_','ln_','swiglu_','small_','direct_',
            'attention_','patch_','vision_','gelu_','terminal_'))}

    def post_attention(self,h,attn,layer):
        f=self.features
        if 'rms_warps' in f or 'rms_fusion' in f:
            return kernels.add_rms(h,attn,None,layer.post_attention_layernorm,warps=f.get('rms_warps',4),fusion=f.get('rms_fusion',False))
        return super().post_attention(h,attn,layer)

    def carry(self,a,b,deep,module,start,nv):
        f=self.features
        if 'rms_warps' in f or 'rms_fusion' in f:
            return kernels.add_rms(a,b,deep,module,start,nv,f.get('rms_warps',4),f.get('rms_fusion',False))
        return super().carry(a,b,deep,module,start,nv)

    def linear(self,x,w):
        f=self.features
        if f.get('small_linear') and x.numel()//x.shape[-1]<=16:
            return kernels.small_linear(x,w,f.get('small_bn',64),f.get('small_bk',64),f.get('small_split',1),f.get('small_warps',4))
        return F.linear(x,w)

    def mlp(self,module,w,x):
        f=self.features
        if not (f.get('swiglu_rows') or f.get('small_linear')):return super().mlp(module,w,x)
        assert not self.fp8_mlp
        packed,down=self.narrow.get(id(module),(w,module.down_proj.weight))
        gate,up=self.linear(x,packed).chunk(2,dim=-1)
        if f.get('swiglu_rows'):
            z=kernels.swiglu_rows(gate,up,f.get('mlp_block',512),f['swiglu_rows'],f.get('mlp_warps',4),
                f.get('swiglu_fast',False),f.get('swiglu_fusion',False))
        else:z=swiglu(gate,up,f.get('mlp_block',512),f.get('mlp_warps',4))
        return self.linear(z,down)

    def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
        f=self.features
        if not any(k in f for k in ('qk_rows','qk_half','qk_warps','qk_pad','qk_fusion','direct_gqa','attention_backend','attention_triton','projection_split')):
            return super().attention(a,w,hidden_states,position_embeddings,attention_mask,**kwargs)
        assert attention_mask is None and not self.fp8_attention and not self.pruned_attention
        shape=hidden_states.shape[:-1];head_shape=(*shape,-1,a.head_dim)
        if f.get('projection_split'):
            # Packing is restored only for the norm kernel's checked shared stride.
            packed=torch.cat((a.q_proj(hidden_states),a.k_proj(hidden_states),a.v_proj(hidden_states)),dim=-1)
        else:packed=F.linear(hidden_states,w)
        q,k,v=packed.split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],-1)
        q,k=kernels.qk_pair(q.view(head_shape),k.view(head_shape),a.q_norm,a.k_norm,*position_embeddings,
            f.get('qk_rows',4),f.get('qk_warps',4),f.get('qk_half',False),f.get('qk_pad',1),f.get('qk_fusion',False))
        v=v.view(head_shape).transpose(1,2)
        if f.get('attention_triton'):
            out=kernels.attention(q,k,v,a.scaling,f.get('attn_bm',16),f.get('attn_bn',32),f.get('attn_warps',4))
        else:
            with backend(f.get('attention_backend')):
                if f.get('direct_gqa'):
                    out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True,scale=a.scaling,enable_gqa=True).transpose(1,2).contiguous()
                else:out=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,v,None,dropout=0.,scaling=a.scaling)[0]
        return a.o_proj(out.reshape(*shape,-1).contiguous()),None

    def terminal(self,layer,qw,gw,h,normalized,rope,picks):
        f=self.features
        if not any(f.get(k) for k in ('terminal_gqa','terminal_fused_rope','small_linear')):
            return super().terminal(layer,qw,gw,h,normalized,rope,picks)
        a=layer.self_attn;t=normalized.shape[1];d=a.head_dim
        selected=picks if self.readout=='head' else picks[8:9]
        narrow=normalized[0].index_select(0,selected)
        q=self.linear(narrow,a.q_proj.weight).view(1,-1,a.q_proj.out_features//d,d)
        k,v=F.linear(normalized,self.terminal_kv).chunk(2,-1)
        k=k.view(1,t,-1,d);v=v.view(1,t,-1,d).transpose(1,2)
        cos,sin=rope;qc=cos.index_select(1,selected);qs=sin.index_select(1,selected)
        if f.get('terminal_fused_rope'):
            q=norm_rope(q.contiguous(),a.q_norm,qc,qs);k=norm_rope(k.contiguous(),a.k_norm,cos,sin)
        else:
            q=norm_rotary(q.contiguous(),a.q_norm,qc,qs);k=norm_rotary(k.contiguous(),a.k_norm,cos,sin)
        group=a.q_proj.out_features//a.k_proj.out_features
        if not f.get('terminal_gqa'):k=k.repeat_interleave(group,1);v=v.repeat_interleave(group,1)
        allowed=(torch.arange(t,device=h.device)[None,:]<=selected[:,None])[None,None]
        with backend(f.get('terminal_backend')):
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.,is_causal=False,scale=a.scaling,
                enable_gqa=bool(f.get('terminal_gqa')))
        attn=self.linear(out.transpose(1,2).reshape(len(selected),-1).contiguous(),a.o_proj.weight)
        residual,post=self.post_attention(h[0].index_select(0,selected),attn,layer)
        return residual+self.mlp(layer.mlp,gw,post)
