"""Explicit visual adaptation contract; official heads and calibration are unchanged."""
import importlib.util,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).parent

def source_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);__import__('sys').modules[name]=m;spec.loader.exec_module(m);return m
HEADS=source_module('fundm_official_heads',ROOT/'official/FunDM-main/scripts/heads.py')
PERM=source_module('anyjev_official_permute',ROOT/'official/AnyJev-main/anyjev/calibrate/permute.py')
CONTEXT=source_module('anyjev_official_contextual',ROOT/'official/AnyJev-main/anyjev/calibrate/contextual.py')

class MidHead(nn.Module):
 def __init__(self,hidden):super().__init__();self.head=HEADS.ComparatorHead(hidden,128)
 def score(self,hq,hk,mask):
  def rms(x):x=x.float();return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)
  return self.head.score(rms(hq),rms(hk),mask)

def kd_loss(teacher,student):
 pt=torch.softmax(teacher,-1);lt=torch.log_softmax(teacher,-1);ls=torch.log_softmax(student,-1)
 return (pt*(lt.clamp_min(-1e4)-ls.clamp_min(-1e4))).masked_fill(pt==0,0).sum(-1).mean()

def option_positions(m,g,q):
 from shunong_accel._legacy.vdm.models.prompts import suffix_text, LETTERS
 text=suffix_text(q['instruction'],q['candidates'],q['qtype']);enc=m.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 assert enc['input_ids']==g.suffix_ids[0].tolist()
 cursor=text.index('\nOptions:\n')+len('\nOptions:\n');ends=[]
 for i,opt in enumerate(q['candidates']):
  line=f'{LETTERS[i]}. {opt}';assert text[cursor:cursor+len(line)]==line
  last=cursor+len(line.rstrip())-1;matches=[j for j,(a,b) in enumerate(enc['offset_mapping']) if a<=last<b];assert len(matches)==1
  ends.append(g.prefix_len+matches[0]);cursor+=len(line)+1
 assert len(set(ends))==len(ends) and max(ends)<g.prefix_len+len(g.suffix_ids[0])-1
 return ends

def reference_distribution(position_probs,perms,prior=None,strength=.75):
 rows=[]
 for ps,perm in zip(position_probs,perms):
  p=[max(float(x),1e-300) for x in ps]
  if prior is not None:p=[max(x/max(float(y),1e-8)**strength,1e-8) for x,y in zip(p,prior)]
  z=sum(p);p=[x/z for x in p];row=[0.]*len(p)
  for j,i in enumerate(perm):row[i]=p[j]
  rows.append(row)
 logp=[sum(math.log(max(row[i],1e-12)) for row in rows)/len(rows) for i in range(len(rows[0]))];peak=max(logp);out=[math.exp(x-peak) for x in logp];s=sum(out);return [x/s for x in out]

def aggregate(position_probs,perms,prior=None):
 p=np.asarray(position_probs,dtype=np.float64)
 if prior is not None:p=CONTEXT.apply_contextual(p,np.asarray(prior)**.75)
 return PERM.marginalize(p,perms,'logmean')

def golden():
 for k in [2,3,4,5,6,8]:
  perms=PERM.cyclic_shifts(k);p=np.arange(1,k*k+1,dtype=float).reshape(k,k);p/=p.sum(-1,keepdims=True);prior=np.arange(1,k+1,dtype=float);prior/=prior.sum()
  for pr in [None,prior]:np.testing.assert_allclose(aggregate(p,perms,pr),reference_distribution(p,perms,pr),rtol=1e-12,atol=1e-12)
  assert perms==[[(j+s)%k for j in range(k)] for s in range(k)]
 t=torch.tensor([[1.,2.,3.],[4.,2.,float('-inf')]]);s=torch.tensor([[3.,1.,2.],[1.,4.,float('-inf')]])
 ref=[]
 for a,b in zip(t.tolist(),s.tolist()):
  valid=[i for i,v in enumerate(a) if math.isfinite(v)];ta=[math.exp(a[i]-max(a[j] for j in valid)) for i in valid];sa=[math.exp(b[i]-max(b[j] for j in valid)) for i in valid];ta=[v/sum(ta) for v in ta];sa=[v/sum(sa) for v in sa];ref.append(sum(x*math.log(x/y) for x,y in zip(ta,sa)))
 assert abs(kd_loss(t,s).item()-sum(ref)/len(ref))<1e-6
 return dict(passed=True,rotations_remapping_prior_geometric_mean=True,teacher_student_KL_math_reference=True)
