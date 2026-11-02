"""Frozen family interface using tested runtime, without INT4 residency."""
import gc
import torch
from .night_runtime import Runtime as Night
from .depth_runtime import Runtime as Rectangular
from ..api import sha256 as sha
import json
from pathlib import Path
def read(path):return json.loads(Path(path).read_text())

class Runtime(Night):
 def pack_release(self,profile):
  result=Rectangular.pack_release(self,profile)
  self.banks={'fp8':self.quant}
  assert torch.count_nonzero(self.repair.up)==0 and torch.count_nonzero(self.repair.bias)==0
  return result

class Accelerator:
 def __init__(self,config,fast=False):
  self.config=config;self.rt=Runtime(config);self.fast=fast
  state=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)['state_dict']
  assert torch.count_nonzero(state['up'])==0 and torch.count_nonzero(state['bias'])==0
  self.rt.repair.load_state_dict(state)
  if fast:self.activate_fast()
 @property
 def model(self):return self.rt.model
 def activate_fast(self):
  c=self.config;assert sha(c['profile_path'])==c['profile_sha256']
  self.rt.profiles=read(c['profile_path'])['profiles'];self.rt.select(c['variant'])
  memory=self.rt.pack_release(self.rt.profiles[c['variant']]);self.rt.set_mode(c['execution_mode']);self.rt.select(c['variant']);self.fast=True
  return memory
 def predict(self,image_path,question,candidates,item_id='request',backend='fast'):
  assert 2<=len(candidates)<=8 and all(isinstance(x,str) and x for x in candidates)
  row=dict(image_path=image_path,instruction=question,candidates=candidates,item_id=item_id)
  if backend=='native':
   if self.fast:raise RuntimeError('BF16 weights released')
   logits=self.rt.native(row);captured=False
  else:
   assert backend=='fast' and self.fast
   g=self.rt.prepare(row);z,captured=self.rt.graph(g);logits=z.cpu().tolist()
  prediction=max(range(len(logits)),key=lambda i:(logits[i],-i))
  return dict(id=item_id,answer=dict(letter=chr(65+prediction),text=candidates[prediction]),logits=logits),captured
