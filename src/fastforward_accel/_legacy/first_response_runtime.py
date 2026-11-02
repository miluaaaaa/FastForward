"""Prebuilt full36 length buckets; request path never captures a CUDA graph."""
import time
import torch
from fastforward_accel._legacy.packing_layout_runtime import PackedGraph24, pack_weights
from fastforward_accel._legacy.serving_dynamic_norm import fused_norm

CAPACITIES=(256,384,512,768,1024,1536,2048)
def capacity(n):return next((b for b in CAPACITIES if b>=n),None)

class FirstResponse36(PackedGraph24):
    def __init__(self,m,head,packed,style,capacities=CAPACITIES):
        super().__init__(m,head,packed+pack_weights(tuple(m.vl_model.language_model.layers[24:])),qkv=True,gateup=True)
        assert style in ['implicit','explicit'];self.style=style;self.capacities=tuple(capacities)
    def pad(self,args,b):
        x,ds,pos,picks,mask,_=args;n=x.shape[1];assert n<=b
        xx=torch.zeros((1,b,x.shape[-1]),device=x.device,dtype=x.dtype);xx[:,:n].copy_(x)
        dd=torch.zeros((len(ds),1,b,x.shape[-1]),device=x.device,dtype=x.dtype);dd[:,:,:n].copy_(ds)
        pp=torch.zeros((3,1,b),device=x.device,dtype=pos.dtype);pp[:,:,:n].copy_(pos)
        attention=torch.empty(0,device=x.device,dtype=torch.bool)
        if self.style=='explicit':
            attention=torch.ones((b,b),device=x.device,dtype=torch.bool).tril()[None,None]
            attention[:,:,:,n:]=False
        return xx,dd,pp,picks.clone(),mask.clone(),attention
    def core(self,args):
        h,ds,pos,picks,_,attention=args;lm=self.m.vl_model.language_model
        rope=lm.rotary_emb(h,pos);cache_position=torch.arange(h.shape[1],device=h.device)
        attention=attention if self.style=='explicit' and attention.numel() else None
        for j,(layer,(qw,gw)) in enumerate(zip(lm.layers,self.packed)):
            attn,_=self.attention(layer.self_attn,qw,hidden_states=fused_norm(h,layer.input_layernorm),position_embeddings=rope,attention_mask=attention,position_ids=pos[0],cache_position=cache_position,use_cache=False)
            h=h+attn;h=h+self.mlp(layer.mlp,gw,fused_norm(h,layer.post_attention_layernorm))
            if j<len(ds):h=h+ds[j]
        # Read the actual final prompt token, never the last padding token.
        last=lm.norm(h[0].index_select(0,picks[8:9])).to(self.m.dtype)
        return self.m.lm_head(last).index_select(-1,self.m.option_token_ids).float()
    def reference(self,args):
        h,ds,pos,picks,_,attention=args;lm=self.m.vl_model.language_model
        rope=lm.rotary_emb(h,pos);cache_position=torch.arange(h.shape[1],device=h.device)
        attention=attention if self.style=='explicit' and attention.numel() else None
        for j,layer in enumerate(lm.layers):
            h=layer(h,position_embeddings=rope,attention_mask=attention,position_ids=pos[0],cache_position=cache_position,use_cache=False)
            if j<len(ds):h=h+ds[j]
        last=lm.norm(h[0].index_select(0,picks[8:9])).to(self.m.dtype)
        return self.m.lm_head(last).index_select(-1,self.m.option_token_ids).float()
    def prebuild(self,prototype):
        start=time.perf_counter();n=prototype[0].shape[1]
        for b in self.capacities:
            # A real model prompt prefix supplies preparation inputs for short buckets.
            clipped=list(prototype);limit=min(n,b)
            clipped[0]=clipped[0][:,:limit];clipped[1]=clipped[1][:,:,:limit];clipped[2]=clipped[2][:,:,:limit]
            clipped[3]=clipped[3].clamp_max(limit-1)
            self.build(self.pad(tuple(clipped),b))
        torch.cuda.synchronize();return time.perf_counter()-start
    def inputs_fast(self,g):
        m=self.m;ids=g.full_ids[None].cuda();vi=(g.full_ids==m.image_token_id).nonzero().flatten().cuda()
        visual,deep=m.vl_model.get_image_features(g.pixel_values.to('cuda',m.dtype),g.image_grid_thw.cuda());visual=torch.cat(visual)
        assert len(vi)==len(visual)
        h=m.vl_model.get_input_embeddings()(ids);h[0].index_copy_(0,vi,visual.to(h.dtype))
        ds=torch.zeros((len(deep),*h.shape),device=h.device,dtype=h.dtype)
        for j,t in enumerate(deep):ds[j,0].index_copy_(0,vi,t.to(h.dtype))
        k=len(g.option_ends);p=torch.zeros(9,dtype=torch.long);p[:k]=torch.tensor(g.option_ends);p[8]=len(g.full_ids)-1
        return h,ds,g.position_ids[0][:,None].cuda(),p.cuda(),(torch.arange(8)<k)[None].cuda(),torch.empty(0,device='cuda')
    def run(self,g):
        args=self.inputs_fast(g);b=next((b for b in self.capacities if b>=len(g.full_ids)),None)
        if b is None:return self.core(args),None
        return self.replay(self.pad(args,b)),b
