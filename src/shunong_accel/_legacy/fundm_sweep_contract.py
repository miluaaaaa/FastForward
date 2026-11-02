"""Frozen finite search contract for a visual shallow-decision model."""
import itertools
import torch
from torch import nn
from shunong_accel._legacy.decision_pilot_contract import HEADS, kd_loss, golden
DEPTHS=[4,8,12,16,20,24,28,32,36]
ARCHES=['linear','comparator']
LR_SCALES=[.5,1.,2.]
SEEDS=[0,1]
EPOCHS=30
BATCH=256
class SweepHead(nn.Module):
 def __init__(self,hidden,arch):super().__init__();self.head=HEADS.HEADS[arch](hidden,128)
 def score(self,q,k,mask):
  def rms(x):x=x.float();return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
  return self.head.score(rms(q),rms(k),mask)
def configurations():
 return [dict(id=f'L{L:02d}-{arch}-r{ri}-s{seed}',layer=L,arch=arch,lr=(.001 if arch=='linear' else .0005)*scale,lr_scale=scale,seed=seed,epochs=EPOCHS,batch=BATCH) for L,arch,(ri,scale),seed in itertools.product(DEPTHS,ARCHES,enumerate(LR_SCALES),SEEDS)]
def select_family(rows):
 families={}
 for r in rows:
  key=(r['layer'],r['arch']);best=families.get(key)
  if best is None or (r['validation_kl'],r['id'])<(best['validation_kl'],best['id']):families[key]=r
 return [families[(L,A)] for L in DEPTHS for A in ARCHES]
def shortlist(candidates):
 """Use only gold-free validation agreement and real latency; never test labels."""
 chosen=[]
 def add(r,why):
  if r and r['id'] not in [x['id'] for x in chosen]:chosen.append(dict(r,reasons=[why]))
  elif r:next(x for x in chosen if x['id']==r['id'])['reasons'].append(why)
 for agreement in [.99,.98,.95,.90]:
  options=[r for r in candidates if r['validation_agreement']>=agreement]
  add(min(options,key=lambda r:(r['timing_ms'],r['validation_kl'],r['id'])) if options else None,f'fastest_validation_agreement_at_least_{agreement}')
 add(min(candidates,key=lambda r:(r['validation_kl'],r['timing_ms'],r['id'])),'minimum_validation_KL')
 return chosen
if __name__=='__main__':
 assert len(configurations())==108 and len({r['id'] for r in configurations()})==108
 fake=[dict(c,validation_kl=float(36-c['layer'])+.01*c['seed']) for c in configurations()];assert len(select_family(fake))==18
 print(golden())
