"""Independent private runtime candidates; no shared package monkey-patches."""
import contextlib
import bisect
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision,ALL_ATTENTION_FUNCTIONS
from fastforward_accel._legacy.serving_dynamic_norm import fused_norm
from fastforward_accel._legacy.sub10_runtime import fast_norm, text_parts
from fastforward_accel._legacy.search_transport import Packet
from fastforward_accel._legacy.input_transport import PinnedTransport
from fastforward_accel._legacy.rotary_exact_runtime import norm_rotary
from fastforward_accel._legacy.vision_graph_runtime import VisionGraph
from fastforward_accel._legacy.pipeline_runtime import BlockJoint
from fastforward_accel._legacy.search_kernels import swiglu, norm_rope_pair, vision_rope


class SearchVision(VisionGraph):
    def core(self,pixels,geometry):
        f=self.owner.features
        v=self.v;pos,cos,sin=geometry
        if f.get('patch_channels_last'):
            pe=v.patch_embed
            z=pixels.view(-1,pe.in_channels,pe.temporal_patch_size,pe.patch_size,pe.patch_size)
            z=z.to(dtype=pe.proj.weight.dtype).contiguous(memory_format=torch.channels_last_3d)
            h=pe.proj(z).view(-1,pe.embed_dim)+pos
        else:
            h=v.patch_embed(pixels)+pos
        n=h.shape[0];h=h.reshape(n,-1);deep=[]
        for j,block in enumerate(v.blocks):
            if hasattr(self,'active_indices') and j not in self.active_indices:continue
            x=block.norm1(h);attn=block.attn
            q,k,value=attn.qkv(x).reshape(n,3,attn.num_heads,-1).permute(1,0,2,3).unbind(0)
            if f.get('vision_rope'):
                q,k=vision_rope(q,k,cos,sin,f.get('vision_block',512))
            else:
                q,k=apply_rotary_pos_emb_vision(q,k,cos,sin)
            q=q.transpose(0,1).unsqueeze(0);k=k.transpose(0,1).unsqueeze(0);value=value.transpose(0,1).unsqueeze(0)
            out=ALL_ATTENTION_FUNCTIONS['sdpa'](attn,q,k,value,attention_mask=None,scaling=attn.scaling,dropout=0.,is_causal=False)[0]
            if f.get('vision_no_cat'):
                out=out.reshape(n,-1).contiguous()
            else:
                out=torch.cat([out],dim=1).reshape(n,-1).contiguous()
            h=h+attn.proj(out);h=h+block.mlp(block.norm2(h))
            if j in v.deepstack_visual_indexes:
                deep.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(j)](h))
        result=v.merger(h),deep
        if self.owner.validate_layers:
            reference=super().core(pixels,geometry)
            assert torch.equal(result[0],reference[0]),'vision_output'
            assert len(result[1])==len(reference[1])
            assert all(torch.equal(a,b) for a,b in zip(result[1],reference[1])),'vision_deepstack'
            self.owner.operator_checks.append(dict(kind='vision',exact=True))
        return result


class SearchJoint(BlockJoint):
    def __init__(self,*args,features=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.features=features or {}
        self.operator_checks=[]
        self.packets={}
        self.building_packets=False
        self.transport=None
        self.vg=SearchVision(self.m);self.vg.owner=self

    def configure(self,spec,prototype):
        super().configure(spec,prototype)
        if self.features.get('bucket_step'):
            # Keep the original fallback capacity. Finer padding changes GEMM
            # shapes, so it must pass historical exactness as a separate trial.
            step=self.features['bucket_step']
            assert step in (8,16)
            minimum=self.features['bucket_min'];limit=self.features['bucket_limit']
            assert minimum%step==limit%step==0 and minimum>len(self.prefix)
            caps=sorted(set(range(minimum,limit+1,step))|{b for b in self.pos if b>limit})
            self.pos={}
            image=self.m.image_token_id
            for b in caps:
                ids=torch.cat((self.prefix,torch.zeros(b-len(self.prefix),dtype=torch.long)))[None]
                pos,_=self.m.vl_model.get_rope_index(ids,mm_token_type_ids=(ids==image).int(),image_grid_thw=self.grid_cpu)
                self.pos[b]=pos.cuda()
        self.prefix_array=self.prefix.numpy().copy()
        if self.features.get('single_norm'):
            self.norm=fast_norm

    def prepare(self,row,backend):
        if not self.features.get('prepare_fast'):
            return super().prepare(row,backend)
        import fastforward_accel._legacy.sub10_runtime as sub10_runtime
        start=time.perf_counter_ns()
        rgb=sub10_runtime.decode(row['image_path'],self.side,backend)
        decoded=time.perf_counter_ns()
        prefix,suffix=text_parts(row,self.style)
        enc=self.m.tokenizer(suffix,add_special_tokens=False,return_offsets_mapping=True)
        offsets=enc['offset_mapping'];starts=[a for a,b in offsets]
        ids=np.concatenate((self.prefix_array,np.asarray(enc['input_ids'],dtype=np.int64)))
        picks=np.zeros(9,dtype=np.int64)
        cursor=suffix.index('\nOptions:\n')+len('\nOptions:\n')
        for i,opt in enumerate(row['candidates']):
            line=f'{chr(65+i)}. {opt}'
            assert suffix[cursor:cursor+len(line)]==line
            last=cursor+len(line.rstrip())-1
            j=bisect.bisect_right(starts,last)-1
            assert j>=0 and offsets[j][0]<=last<offsets[j][1]
            assert j==0 or offsets[j-1][1]<=last
            picks[i]=len(self.prefix_array)+j
            cursor+=len(line)+1
        picks[8]=len(ids)-1
        return dict(rgb=rgb,ids=torch.from_numpy(ids),picks=torch.from_numpy(picks),k=len(row['candidates']),
            prompt_text=prefix+suffix,decoded_ns=decoded,prepare_ns=time.perf_counter_ns(),start_ns=start)

    def args(self,g,b):
        if not self.building_packets:
            return super().args(g,b)
        packet=Packet([(self.side,self.side,3),(1,b),(9,),(1,8)],
                      [torch.uint8,torch.int64,torch.int64,torch.bool])
        packet.fill(g);self.packets[b]=packet
        return packet.device_views

    def prebuild(self,g):
        self.building_packets=self.features.get('packet',False)
        try:
            seconds=super().prebuild(g)
        finally:
            self.building_packets=False
        if self.features.get('pinned'):
            self.transport=PinnedTransport(self)
        return seconds

    def run_joint(self,g):
        if self.transport is not None:
            return self.transport.run(g)
        if not self.features.get('packet'):
            return super().run_joint(g)
        b=next((b for b in self.cache if b>=len(g['ids'])),None)
        if b is None:raise RuntimeError('No prepared packet bucket')
        self.packets[b].fill(g)
        static,graph,out=self.cache[b];graph.replay()
        return out,True

    def carry(self,a,b,deep,module,start,nv):
        if not self.features.get('single_norm'):
            return super().carry(a,b,deep,module,start,nv)
        residual=self.inject_reference(a+b,deep)
        out=fast_norm(residual,module)
        if self.validate_layers:
            assert torch.equal(out,fused_norm(residual,module)),'single_norm_carry'
        return residual,out

    def post_attention(self,h,attn,layer):
        if not self.features.get('single_norm'):
            return super().post_attention(h,attn,layer)
        residual=h+attn
        out=fast_norm(residual,layer.post_attention_layernorm)
        if self.validate_layers:
            assert torch.equal(out,fused_norm(residual,layer.post_attention_layernorm)),'single_norm_post'
        return residual,out

    def mlp(self,m,w,x):
        f=self.features
        if not (f.get('mlp_fuse') or f.get('mlp_split')):
            return super().mlp(m,w,x)
        if f.get('mlp_split'):
            gate,up=m.gate_proj(x),m.up_proj(x)
        else:
            gate,up=F.linear(x,w).chunk(2,dim=-1)
        product=swiglu(gate,up,f.get('mlp_block',1024),f.get('mlp_warps',4)) if f.get('mlp_fuse') else m.act_fn(gate)*up
        out=m.down_proj(product)
        if self.validate_layers:
            g,u=F.linear(x,w).chunk(2,dim=-1)
            expected=m.act_fn(g)*u
            assert torch.equal(product,expected),'mlp_product'
            assert torch.equal(out,m.down_proj(expected)),'mlp_down'
            self.operator_checks.append(dict(kind='mlp',exact=True,shape=list(product.shape)))
        return out

    def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
        f=self.features
        backend=f.get('attention_backend')
        context=sdpa_kernel(getattr(SDPBackend,backend)) if backend else contextlib.nullcontext()
        if not (f.get('qk_strided') or f.get('qkv_split')):
            with context:
                return super().attention(a,w,hidden_states=hidden_states,position_embeddings=position_embeddings,attention_mask=attention_mask,**kwargs)
        shape=hidden_states.shape[:-1];head_shape=(*shape,-1,a.head_dim)
        if f.get('qkv_split'):
            q,k,v=a.q_proj(hidden_states),a.k_proj(hidden_states),a.v_proj(hidden_states)
            if self.validate_layers:
                rq,rk,rv=F.linear(hidden_states,w).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],dim=-1)
                assert all(torch.equal(x,y) for x,y in [(q,rq),(k,rk),(v,rv)]),'qkv_split'
        else:
            q,k,v=F.linear(hidden_states,w).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],dim=-1)
        q0,k0=q.view(head_shape),k.view(head_shape)
        if f.get('qk_strided'):
            q,k=norm_rope_pair(q0,k0,a.q_norm,a.k_norm,*position_embeddings,
                                joint=f.get('qk_joint',False),block=f.get('qk_block',512))
        else:
            q=norm_rotary(q0.contiguous(),a.q_norm,*position_embeddings)
            k=norm_rotary(k0.contiguous(),a.k_norm,*position_embeddings)
        if self.validate_layers:
            rq=norm_rotary(q0.contiguous(),a.q_norm,*position_embeddings)
            rk=norm_rotary(k0.contiguous(),a.k_norm,*position_embeddings)
            assert torch.equal(q,rq) and torch.equal(k,rk),'qk_norm_rope'
            self.operator_checks.append(dict(kind='qk',exact=True))
        v=v.view(head_shape).transpose(1,2)
        with context:
            out,_=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,v,attention_mask,dropout=0.,scaling=a.scaling)
        return a.o_proj(out.reshape(*shape,-1).contiguous()),None
