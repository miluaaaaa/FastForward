"""Single suffix tokenization and direct fill of existing language graph buffers."""
import torch
from shunong_accel._legacy.vdm.models.prompts import prefix_text, suffix_text, LETTERS
from shunong_accel._legacy.vdm.models.vdm_model import PreparedGroup

class InputFast:
 def __init__(self,m,rt):self.m=m;self.rt=rt
 def prepare(self,image,q):
  m=self.m;enc=m.processor(text=[prefix_text('')],images=[image],return_tensors='pt');prefix=enc['input_ids'][0];text=suffix_text(q['instruction'],q['candidates'],q['qtype']);tok=m.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True);suffix=torch.tensor(tok['input_ids'],dtype=torch.long);full=torch.cat((prefix,suffix));mm=(full[None]==m.image_token_id).to(torch.int32)
  pos,_=m.vl_model.get_rope_index(full[None].cuda(),mm_token_type_ids=mm.cuda(),image_grid_thw=enc['image_grid_thw'].cuda())
  cursor=text.index('\nOptions:\n')+len('\nOptions:\n');ends=[]
  for i,opt in enumerate(q['candidates']):
   line=f'{LETTERS[i]}. {opt}';assert text[cursor:cursor+len(line)]==line;last=cursor+len(line.rstrip())-1;matches=[j for j,(a,b) in enumerate(tok['offset_mapping']) if a<=last<b];assert len(matches)==1;ends.append(len(prefix)+matches[0]);cursor+=len(line)+1
  assert len(set(ends))==len(ends) and max(ends)<len(full)-1
  g=PreparedGroup(prefix,[suffix],enc['pixel_values'],enc['image_grid_thw'],[pos[:,0]],[q['qtype']],[len(ends)],m.image_token_id);g.full_ids=full;g.option_ends=ends;return g
 def run(self,g):
  m=self.m;n=len(g.full_ids);static,graph,out=self.rt.cache[n];x,ds,pos,picks,mask,_=static
  visual,deep=m.vl_model.get_image_features(g.pixel_values.to('cuda',m.dtype),g.image_grid_thw.cuda());vi=(g.full_ids==m.image_token_id).nonzero().flatten().cuda();visual=torch.cat(visual);assert len(vi)==len(visual)
  x.copy_(m.vl_model.get_input_embeddings()(g.full_ids[None].cuda()));x[0].index_copy_(0,vi,visual.to(x.dtype));ds.zero_()
  for j,t in enumerate(deep):ds[j,0].index_copy_(0,vi,t.to(ds.dtype))
  pos.copy_(g.position_ids[0][:,None]);k=len(g.option_ends);cpu_picks=torch.zeros(9,dtype=torch.long);cpu_picks[:k]=torch.tensor(g.option_ends);cpu_picks[8]=n-1;picks.copy_(cpu_picks);mask.copy_((torch.arange(8)<k)[None]);graph.replay();return out
