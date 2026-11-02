"""Continue retained-score execution optimization in a private runtime."""
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS
from shunong_accel._legacy.aggressive_runtime import AggressiveJoint
from shunong_accel._legacy.search_runtime import SearchVision
from shunong_accel._legacy.search_kernels import vision_rope
from shunong_accel._legacy.rotary_exact_runtime import norm_rotary
from shunong_accel._legacy.lower_kernels import add_rms, add_layernorm, norm_rope, norm_rope_pair
from shunong_accel._legacy.search_transport import Packet


class SerialPacket(Packet):
    """Staging for the serial CPU-returning API; the prior call has completed.

    The public Decision lock and blocking logits.cpu() establish ownership.
    This packet must not be used for asynchronous or overlapping requests.
    """
    def __init__(self,shapes,dtypes):
        super().__init__(shapes,dtypes)
        self.numpy_views=tuple(x.numpy() for x in self.host_views)

    def fill(self,g):
        rgb,ids,picks,mask=self.numpy_views
        rgb[:]=g['rgb'];ids.fill(0);ids[0,:len(g['ids'])]=g['ids'].numpy()
        picks[:]=g['picks'].numpy();mask.fill(False);mask[0,:g['k']]=True
        self.device.copy_(self.host,non_blocking=True)


class LowerVision(SearchVision):
    def core(self,pixels,geometry):
        if not self.owner.features.get('vision_add_norm'):return super().core(pixels,geometry)
        v=self.v;pos,cos,sin=geometry
        h=v.patch_embed(pixels)+pos;h=h.reshape(h.shape[0],-1);n=h.shape[0];deep=[]
        indices=[j for j in range(len(v.blocks)) if j in self.active_indices]
        x=v.blocks[indices[0]].norm1(h)
        for offset,j in enumerate(indices):
            block=v.blocks[j];attn=block.attn
            q,k,value=attn.qkv(x).reshape(n,3,attn.num_heads,-1).permute(1,0,2,3).unbind(0)
            q,k=vision_rope(q,k,cos,sin,self.owner.features.get('vision_block',512))
            q=q.transpose(0,1).unsqueeze(0);k=k.transpose(0,1).unsqueeze(0);value=value.transpose(0,1).unsqueeze(0)
            out=ALL_ATTENTION_FUNCTIONS['sdpa'](attn,q,k,value,attention_mask=None,scaling=attn.scaling,dropout=0.,is_causal=False)[0]
            h,post=add_layernorm(h,attn.proj(out.reshape(n,-1).contiguous()),block.norm2)
            mlp=block.mlp(post)
            if offset+1<len(indices):h,x=add_layernorm(h,mlp,v.blocks[indices[offset+1]].norm1)
            else:h=h+mlp
            if j in v.deepstack_visual_indexes:
                deep.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(j)](h))
        return v.merger(h),deep


class LowerJoint(AggressiveJoint):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.vg=LowerVision(self.m);self.vg.owner=self

    def configure(self,spec,prototype):
        super().configure(spec,prototype)
        self.spec['lower_operators']={k:v for k,v in self.features.items() if k in
            ['fused_add_norm','onepass_qk','vision_add_norm','terminal_queries','lean_call','serial_packet','bucket_step']}
        if self.features.get('terminal_queries'):
            a=self.m.vl_model.language_model.layers[self.language_indices[-1]].self_attn
            self.terminal_kv=torch.cat((a.k_proj.weight,a.v_proj.weight)).contiguous()

    def args(self,g,b):
        if self.building_packets and self.features.get('serial_packet'):
            assert self.features.get('lean_call'),'Serial packet requires the synchronous public API'
            packet=SerialPacket([(self.side,self.side,3),(1,b),(9,),(1,8)],
                [torch.uint8,torch.int64,torch.int64,torch.bool])
            packet.fill(g);self.packets[b]=packet
            return packet.device_views
        return super().args(g,b)

    def post_attention(self,h,attn,layer):
        if self.features.get('fused_add_norm'):return add_rms(h,attn,None,layer.post_attention_layernorm)
        return super().post_attention(h,attn,layer)

    def carry(self,a,b,deep,module,start,nv):
        if self.features.get('fused_add_norm'):return add_rms(a,b,deep,module,start,nv)
        return super().carry(a,b,deep,module,start,nv)

    def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
        if not self.features.get('onepass_qk'):return super().attention(a,w,hidden_states,position_embeddings,attention_mask,**kwargs)
        assert not self.pruned_attention and not self.fp8_attention
        shape=hidden_states.shape[:-1];head_shape=(*shape,-1,a.head_dim)
        q,k,v=F.linear(hidden_states,w).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],dim=-1)
        cos,sin=position_embeddings
        mode=self.features['onepass_qk']
        if mode in ('rows4','rows8'):
            q,k=norm_rope_pair(q.view(head_shape),k.view(head_shape),a.q_norm,a.k_norm,cos,sin,int(mode[4:]))
        else:
            q=norm_rope(q.view(head_shape),a.q_norm,cos,sin)
            k=norm_rope(k.view(head_shape),a.k_norm,cos,sin)
        v=v.view(head_shape).transpose(1,2)
        out,_=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,v,attention_mask,dropout=0.,scaling=a.scaling)
        return a.o_proj(out.reshape(*shape,-1).contiguous()),None

    def terminal(self,layer,qw,gw,h,normalized,rope,picks):
        """Keep complete KV context and the original causal mask for each query."""
        a=layer.self_attn;t=normalized.shape[1];d=a.head_dim
        selected=picks if self.readout=='head' else picks[8:9]
        narrow=normalized[0].index_select(0,selected)
        q=F.linear(narrow,a.q_proj.weight).view(1,-1,a.q_proj.out_features//d,d)
        k,v=F.linear(normalized,self.terminal_kv).chunk(2,dim=-1)
        k=k.view(1,t,-1,d);v=v.view(1,t,-1,d).transpose(1,2)
        cos,sin=rope;qc=cos.index_select(1,selected);qs=sin.index_select(1,selected)
        q=norm_rotary(q.contiguous(),a.q_norm,qc,qs)
        k=norm_rotary(k.contiguous(),a.k_norm,cos,sin)
        group=a.q_proj.out_features//a.k_proj.out_features
        k=k.repeat_interleave(group,dim=1);v=v.repeat_interleave(group,dim=1)
        allowed=(torch.arange(t,device=h.device)[None,:]<=selected[:,None])[None,None]
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.,is_causal=False,scale=a.scaling)
        attn=a.o_proj(out.transpose(1,2).reshape(len(selected),-1).contiguous())
        old=h[0].index_select(0,selected)
        residual,post=self.post_attention(old,attn,layer)
        return residual+self.mlp(layer.mlp,gw,post)
