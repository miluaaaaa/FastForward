"""Exact-grid full vision graph; cache geometry/execution only, never image features."""
import time
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision,ALL_ATTENTION_FUNCTIONS

class VisionGraph:
 def __init__(self,m):
  self.v=m.vl_model.visual;self.original=m.vl_model.get_image_features;self.cache={};self.geometry={};self.mode='baseline';self.builds=[];self.last_fallback=False;self.events=None
 def key(self,grid):
  key=tuple(int(x) for x in grid.reshape(-1).tolist());assert len(key)==3 and key[0]==1,'Still-image singleton only';return key
 def geo(self,grid):
  key=self.key(grid)
  if key not in self.geometry:
   pos=self.v.fast_pos_embed_interpolate(grid);rot=self.v.rot_pos_emb(grid);n=pos.shape[0];rot=rot.reshape(n,-1);emb=torch.cat((rot,rot),dim=-1);self.geometry[key]=(pos,emb.cos(),emb.sin())
  return self.geometry[key]
 def core(self,pixels,geometry):
  v=self.v;pos,cos,sin=geometry;h=v.patch_embed(pixels)+pos;n=h.shape[0];h=h.reshape(n,-1);deep=[]
  for j,block in enumerate(v.blocks):
   if hasattr(self,'active_indices') and j not in self.active_indices:continue
   x=block.norm1(h);attn=block.attn
   q,k,value=attn.qkv(x).reshape(n,3,attn.num_heads,-1).permute(1,0,2,3).unbind(0)
   q,k=apply_rotary_pos_emb_vision(q,k,cos,sin);q=q.transpose(0,1).unsqueeze(0);k=k.transpose(0,1).unsqueeze(0);value=value.transpose(0,1).unsqueeze(0)
   # Identical singleton chunk SDPA call as native path; eliminate per-layer GPU lengths.tolist().
   out=ALL_ATTENTION_FUNCTIONS['sdpa'](attn,q,k,value,attention_mask=None,scaling=attn.scaling,dropout=0.,is_causal=False)[0]
   out=torch.cat([out],dim=1).reshape(n,-1).contiguous();h=h+attn.proj(out);h=h+block.mlp(block.norm2(h))
   if j in v.deepstack_visual_indexes:deep.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(j)](h))
  return v.merger(h),deep
 def build(self,pixels,grid):
  key=self.key(grid)
  if key in self.cache:return self.cache[key]
  start=time.perf_counter();geometry=self.geo(grid);static=pixels.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):self.core(static,geometry)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):out=self.core(static,geometry)
  torch.cuda.synchronize();entry=(static,graph,out);self.cache[key]=entry;self.builds.append(dict(grid=list(key),seconds=time.perf_counter()-start));print('VISION_GRAPH_BUILT',self.builds[-1],flush=True);return entry
 def features(self,pixels,grid,**kwargs):
  assert not kwargs or kwargs.get('return_dict') is None
  if self.events is not None:self.events[0].record()
  self.last_fallback=False
  if self.mode=='baseline':out=self.original(pixels,grid)
  elif self.mode=='static':
   image,deep=self.core(pixels,self.geo(grid));out=((image,),deep)
  else:
   key=self.key(grid)
   if key not in self.cache:self.last_fallback=True;out=self.original(pixels,grid)
   else:
    static,graph,result=self.cache[key];static.copy_(pixels);graph.replay();out=((result[0],),result[1])
  if self.events is not None:self.events[1].record()
  return out
