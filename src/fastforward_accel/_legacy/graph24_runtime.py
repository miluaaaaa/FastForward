"""Fixed first24 decoder layers; original vision reruns each request. No trained parameters."""
import time
import torch
from fastforward_accel._legacy.decision_pilot_contract import option_positions

class Graph24:
 def __init__(self,m,head):
  self.m=m;self.head=head;self.layers=tuple(m.vl_model.language_model.layers[:24]);self.cache={};self.builds=[]
 def inputs(self,g,q):
  m=self.m;ids=torch.cat([g.prefix_ids,g.suffix_ids[0]])[None].cuda();n=ids.shape[1];B=n;H=m.config.text_config.hidden_size
  visual,deep=m.vl_model.get_image_features(g.pixel_values.to('cuda',m.dtype),g.image_grid_thw.cuda());visual=torch.cat(visual,dim=0)
  # Positions resolved from CPU prompt tokens; no GPU boolean indexing or dynamic output sizes.
  vi=(torch.cat([g.prefix_ids,g.suffix_ids[0]])==m.image_token_id).nonzero().flatten().cuda();assert len(vi)==len(visual)
  emb=m.vl_model.get_input_embeddings()(ids);emb[0].index_copy_(0,vi,visual.to(emb.dtype));x=torch.zeros(1,B,H,device='cuda',dtype=emb.dtype);x[:,:n].copy_(emb)
  ds=torch.zeros(len(deep),1,B,H,device='cuda',dtype=emb.dtype)
  for j,t in enumerate(deep):ds[j,0].index_copy_(0,vi,t.to(emb.dtype))
  pos=torch.zeros(3,1,B,device='cuda',dtype=torch.long);pos[:,:,:n].copy_(g.position_ids[0][:,None].cuda())
  opts=option_positions(m,g,q);k=len(opts);picks=torch.zeros(9,device='cuda',dtype=torch.long);picks[:k]=torch.tensor(opts,device='cuda');picks[8]=n-1;mask=torch.arange(8,device='cuda')[None]<k
  mask4=torch.ones(B,B,device='cuda',dtype=torch.bool).tril()[None,None];mask4[:,:,:,n:]=False
  return x,ds,pos,picks,mask,mask4
 def core(self,args):
  x,ds,pos,picks,mask,attention=args;rope=self.m.vl_model.language_model.rotary_emb(x,pos);cache_position=torch.arange(x.shape[1],device='cuda');h=x
  for j,layer in enumerate(self.layers):
   h=layer(h,position_embeddings=rope,attention_mask=None,position_ids=pos[0],cache_position=cache_position,use_cache=False)
   if j<len(ds):h=h+ds[j]
  chosen=h[0].index_select(0,picks)
  return self.head.score(chosen[8][None],chosen[:8][None],mask)
 def build(self,args):
  B=args[0].shape[1]
  if B in self.cache:return self.cache[B]
  start=time.perf_counter();static=tuple(x.clone() for x in args);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):self.core(static)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):out=self.core(static)
  torch.cuda.synchronize();entry=(static,graph,out);self.cache[B]=entry;self.builds.append(dict(bucket=B,seconds=time.perf_counter()-start));print('GRAPH_BUILT',self.builds[-1],flush=True);return entry
 def eager(self,args):return self.core(args)
 def replay(self,args):
  static,graph,out=self.cache[args[0].shape[1]]
  for dst,src in zip(static,args):dst.copy_(src)
  graph.replay();return out
