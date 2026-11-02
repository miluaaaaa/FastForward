"""Frozen-weight GEMM packing; original attention, rounding and scorer semantics."""
import torch
import torch.nn.functional as F
from fastforward_accel._legacy.input_fast_runtime import InputFast
from fastforward_accel._legacy.qk_fused_runtime import QKFusedGraph24
from fastforward_accel._legacy.fused_norm_runtime import fused_norm
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb,ALL_ATTENTION_FUNCTIONS

# Only integer positional metadata runs on CPU. Every model operation stays CUDA.
from fastforward_accel._legacy.vdm.models.prompts import prefix_text, suffix_text, LETTERS
from fastforward_accel._legacy.vdm.models.vdm_model import PreparedGroup
class CPUPositionInput(InputFast):
 def prepare(self,image,q):
  m=self.m;enc=m.processor(text=[prefix_text('')],images=[image],return_tensors='pt');prefix=enc['input_ids'][0];text=suffix_text(q['instruction'],q['candidates'],q['qtype']);tok=m.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True);suffix=torch.tensor(tok['input_ids'],dtype=torch.long);full=torch.cat((prefix,suffix));mm=(full[None]==m.image_token_id).to(torch.int32)
  pos,_=m.vl_model.get_rope_index(full[None],mm_token_type_ids=mm,image_grid_thw=enc['image_grid_thw'])
  cursor=text.index('\nOptions:\n')+len('\nOptions:\n');ends=[]
  for i,opt in enumerate(q['candidates']):
   line=f'{LETTERS[i]}. {opt}';assert text[cursor:cursor+len(line)]==line;last=cursor+len(line.rstrip())-1;matches=[j for j,(a,b) in enumerate(tok['offset_mapping']) if a<=last<b];assert len(matches)==1;ends.append(len(prefix)+matches[0]);cursor+=len(line)+1
  assert len(set(ends))==len(ends) and max(ends)<len(full)-1
  g=PreparedGroup(prefix,[suffix],enc['pixel_values'],enc['image_grid_thw'],[pos[:,0]],[q['qtype']],[len(ends)],m.image_token_id);g.full_ids=full;g.option_ends=ends;return g

def pack_weights(layers):
 result=[]
 for layer in layers:
  a=layer.self_attn;mlp=layer.mlp
  assert all(p.bias is None for p in [a.q_proj,a.k_proj,a.v_proj,mlp.gate_proj,mlp.up_proj])
  result.append((torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]).contiguous(),torch.cat([mlp.gate_proj.weight,mlp.up_proj.weight]).contiguous()))
 return result

class PackedGraph24(QKFusedGraph24):
 def __init__(self,m,head,packed,qkv=False,gateup=False):
  super().__init__(m,head);self.packed=packed;self.qkv=qkv;self.gateup=gateup
 def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
  shape=hidden_states.shape[:-1];head_shape=(*shape,-1,a.head_dim)
  q,k,v=F.linear(hidden_states,w).split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],dim=-1)
  q=fused_norm(q.contiguous().view(head_shape),a.q_norm).transpose(1,2)
  k=fused_norm(k.contiguous().view(head_shape),a.k_norm).transpose(1,2)
  v=v.view(head_shape).transpose(1,2)
  q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
  assert kwargs.get('past_key_values') is None and a.config._attn_implementation=='sdpa'
  out,_=ALL_ATTENTION_FUNCTIONS['sdpa'](a,q,k,v,attention_mask,dropout=0.,scaling=a.scaling)
  return a.o_proj(out.reshape(*shape,-1).contiguous()),None
 def mlp(self,m,w,x):
  gate,up=F.linear(x,w).chunk(2,dim=-1)
  return m.down_proj(m.act_fn(gate)*up)
 def core(self,args):
  patches=[]
  try:
   for layer,(qw,gw) in zip(self.layers,self.packed):
    for module,enabled,fn in [(layer.self_attn,self.qkv,lambda *a,_m=layer.self_attn,_w=qw,**k:self.attention(_m,_w,*a,**k)),(layer.mlp,self.gateup,lambda x,_m=layer.mlp,_w=gw:self.mlp(_m,_w,x))]:
     if enabled:
      patches.append((module,module.__dict__.get('forward')));module.forward=fn
   return super().core(args)
  finally:
   for module,old in reversed(patches):
    if old is None:module.__dict__.pop('forward',None)
    else:module.forward=old
