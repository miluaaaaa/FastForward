"""One-factor, paired LoRA-merge latency/quality experiment on real CUDA."""
import argparse,hashlib,importlib.metadata,json,math,statistics,time
from pathlib import Path
import torch
from PIL import Image
from peft import PeftModel
from safetensors.torch import load_file
from fastforward_accel._legacy.vdm.models.vdm_model import VDM
from fastforward_accel._legacy.visual_jev_compat import install_metadata_shim
from fastforward_accel._legacy.visual_jev_paper_compat import install as install_paper_compat
from fastforward_accel._legacy.eval_visual_jev_paper import complete, logits, prompt_check, qs
from fastforward_accel._legacy.gpu_guard import require_allocation, prove, invalidate

METHODS=['unmerged','merged'];CELLS=[('independent',1),('prefix_share_batch',1),('prefix_share_batch',8),('prefix_share_batch',16),('prefix_share_batch',32)]
def dump(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def target(k):return k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
def exact_adapter(m,checkpoint):
 named=dict(m.backbone.named_parameters())
 assert len([k for k in named if '.lora_' in k])==504
 for k,v in checkpoint.items():assert torch.equal(named[target(k)].detach().cpu(),v.to(named[target(k)].dtype)),k
 assert len(checkpoint)==504 and sum(v.numel() for v in checkpoint.values())==33030144

def load(a):
 m=VDM(str(a.model),device='cuda',dtype=torch.bfloat16,with_heads=False)
 m.processor.image_processor.max_pixels=200704;install_metadata_shim(m);install_paper_compat(m)
 m.backbone=PeftModel.from_pretrained(m.backbone,str(a.adapter),local_files_only=True,is_trainable=False).eval().requires_grad_(False)
 return m

def summarize(rows):
 t=sorted(x['group_ms'] for x in rows);n=rows[0]['N']
 return dict(count=len(rows),group_ms_mean=statistics.mean(t),group_ms_p50=statistics.median(t),group_ms_p95=t[math.ceil(.95*len(t))-1],amortized_ms_per_question=statistics.mean(t)/n,questions_per_second=1000*n/statistics.mean(t))

def main(a):
 require_allocation();a.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(0);torch.set_grad_enabled(False);torch.set_float32_matmul_precision('highest')
 old=json.loads((a.dataset/'inference.json').read_text());fresh=json.loads((a.dataset/'merge_confirmation_inference.json').read_text())
 gold=json.loads((a.dataset/'gold.json').read_text());gold.update(json.loads((a.dataset/'merge_confirmation_gold.json').read_text()))
 groups=old['sentinel_groups']+old['timing_groups']+fresh;assert len(groups)==17 and len({g['image_id'] for g in groups})==17
 for g in groups:
  assert sha(Path(g['image_path']))==g['image_sha256'];assert len(g['questions'])==32 and len({q['question_id'] for q in g['questions']})==32
  for q in g['questions']:
   assert 'label' not in q and 'answer' not in q
   assert gold[q['item_id']]['answer']==q['candidates'][gold[q['item_id']]['index']]
 checkpoint=load_file(a.adapter/'adapter_model.safetensors');models={k:load(a) for k in METHODS}
 for m in models.values():exact_adapter(m,checkpoint)
 # Model-sized copies reside on one allocated physical GPU, in one inference PID.
 # Both are warm-forwarded before semantic results and again in BF16 before timing.
 original_dtypes={n:p.dtype for n,p in models['unmerged'].backbone.named_parameters()}
 buffers={k:{n:b.detach().clone() for n,b in m.vl.named_buffers()} for k,m in models.items()}
 for m in models.values():m.float()
 start=time.perf_counter();models['merged'].backbone=models['merged'].backbone.merge_and_unload(safe_merge=True).eval().requires_grad_(False);torch.cuda.synchronize();merge_s=time.perf_counter()-start
 assert not any('lora_' in n for n,_ in models['merged'].named_parameters())
 first=old['sentinel_groups'][0]
 with Image.open(first['image_path']) as f:image=f.convert('RGB')
 def forward_pair_fp32():
  for m in models.values():complete(m,image,qs(first,1),'prefix_share_batch')
 prove(torch.nn.ModuleList(list(models.values())),forward_pair_fp32,a.output/'fp32_gpu')
 ref=complete(models['unmerged'],image,qs(first,2),'independent');fp32=[]
 for method,m in models.items():
  g=m.prepare_group(image,qs(first,2));prompt_check(m,g,qs(first,2))
  for path in ['independent','prefix_share_batch']:
   z=logits(m,g,path);torch.testing.assert_close(ref,z,rtol=1e-4,atol=.003)
   fp32.append(dict(method=method,path=path,max_logit_diff=float((ref-z).abs().max())))
 print('FP32_MERGE_AND_PATH_PARITY_PASSED',flush=True)
 for method,m in models.items():
  m.bfloat16()
  for n,b in m.vl.named_buffers():b.data=buffers[method][n].to(b.device)
 del buffers
 # Restore all original unmerged LoRA storage precision after the FP32 gate.
 baseline=models['unmerged'];named=dict(baseline.backbone.named_parameters())
 for n,p in named.items():p.data=p.data.to(original_dtypes[n])
 for k,v in checkpoint.items():named[target(k)].copy_(v.to(device=named[target(k)].device,dtype=named[target(k)].dtype))
 exact_adapter(baseline,checkpoint);del checkpoint
 # Independent tensor algebra verifies every merged target in final BF16 storage.
 merged_named=dict(models['merged'].backbone.named_parameters());merged_modules=dict(models['merged'].backbone.named_modules());changed=set();count=0
 for name,module in baseline.backbone.named_modules():
  if not hasattr(module,'lora_A') or 'default' not in module.lora_A:continue
  assert not module.fan_in_fan_out
  canonical=name.removeprefix('base_model.model.')
  A=module.lora_A['default'].weight;B=module.lora_B['default'].weight
  expected=(module.base_layer.weight.float()+(B.float()@A.float())*module.scaling['default']).bfloat16()
  assert torch.equal(merged_modules[canonical].weight,expected),canonical
  changed.add(canonical+'.weight');count+=1
 assert count==252
 for name,p in baseline.backbone.named_parameters():
  if '.lora_' in name:continue
  canonical=name.removeprefix('base_model.model.').replace('.base_layer.','.')
  if canonical not in changed:assert torch.equal(p,merged_named[canonical]),canonical
 del expected,merged_named,merged_modules,named
 versions={k:tuple(p._version for p in m.parameters()) for k,m in models.items()}
 def forward_pair_bf16():
  for m in models.values():complete(m,image,qs(first,32),'prefix_share_batch')
 evidence=prove(torch.nn.ModuleList(list(models.values())),forward_pair_bf16,a.output)
 sentinel=[]
 for group in old['sentinel_groups']:
  with Image.open(group['image_path']) as f:img=f.convert('RGB')
  out={}
  for method,m in models.items():
   g=m.prepare_group(img,qs(group,32));prompt_check(m,g,qs(group,32));out[method]=logits(m,g,'prefix_share_batch')
   # Separately invoke native PEFT/base full forward for the first two items.
   for q in group['questions'][:2]:
    small=m.prepare_group(img,[{k:q[k] for k in ['qtype','instruction','candidates']}]);full=torch.cat([small.prefix_ids,small.suffix_ids[0]])[None].to('cuda');m.vl_model.rope_deltas=None
    native=m.backbone(input_ids=full,pixel_values=small.pixel_values.to('cuda',m.dtype),image_grid_thw=small.image_grid_thw.to('cuda'),position_ids=small.position_ids[0][:,None].to('cuda'),use_cache=False,logits_to_keep=1).logits[0,-1]
    actual=logits(m,small,'independent')[0]
    torch.testing.assert_close(actual,native.index_select(-1,m.option_token_ids).float(),rtol=0,atol=0)
  for i,q in enumerate(group['questions']):
   k=len(q['candidates']);za=out['unmerged'][i,:k];zb=out['merged'][i,:k]
   assert torch.isfinite(za).all() and torch.isfinite(zb).all()
   sentinel.append(dict(item_id=q['item_id'],max_logit_diff=float((za-zb).abs().max()),max_probability_diff=float((za.softmax(-1)-zb.softmax(-1)).abs().max()),flip=int(za.argmax()!=zb.argmax())))
 dump(a.output/'preflight.json',dict(passed=True,adapter_tensors_exact_each=504,adapter_parameters=33030144,merged_targets_exact=252,untargeted_parameters_equal=True,no_remaining_lora_modules=True,fp32=fp32,sentinels=sentinel,native_forward_exact=True,original_prompt_exact=True,merge_seconds_excluded=merge_s,method='Merge LoRA into FP32 base with PEFT safe_merge, then store BF16; final matrices independently verified as BF16(FP32(W)+scale*B@A)',gpu_passed=evidence['passed']))
 print('MERGE_PREFLIGHT_PASSED',flush=True)
 records=[];timings=[]
 def request(group,n,path,method,phase,repeat,index,predfile,timingfile,warmup=False):
  with Image.open(group['image_path']) as f:img=f.convert('RGB')
  questions=qs(group,n);m=models[method]
  torch.cuda.synchronize();start=time.perf_counter();out=complete(m,img,questions,path);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
  if warmup:return
  timing=dict(phase=phase,repeat=repeat,group_index=index,image_id=group['image_id'],N=n,path=path,method=method,group_ms=elapsed)
  timings.append(timing);timingfile.write(json.dumps(timing)+'\n')
  values=out.float().cpu().tolist()
  for i,q in enumerate(group['questions'][:n]):
   z=values[i][:len(q['candidates'])];pick=max(range(len(z)),key=z.__getitem__)
   r=dict(phase=phase,repeat=repeat,group_index=index,image_id=group['image_id'],N=n,path=path,method=method,item_id=q['item_id'],question_id=q['question_id'],logits=z,prediction=pick,gold=gold[q['item_id']]['index'],correct=pick==gold[q['item_id']]['index'])
   records.append(r);predfile.write(json.dumps(r)+'\n')
 with (a.output/'predictions.jsonl').open('w') as predfile,(a.output/'timings.jsonl').open('w') as timingfile:
  for cell,(path,n) in enumerate(CELLS):
   # Match paper: two distinct warm-up images, before paired repeats.
   for i,group in enumerate(old['timing_groups'][:2]):
    for method in METHODS:request(group,n,path,method,'warmup',0,i,predfile,timingfile,warmup=True)
   for repeat in range(3):
    for index,group in enumerate(old['timing_groups'][2:]):
     order=METHODS if (cell+repeat+index)%2==0 else list(reversed(METHODS))
     for method in order:request(group,n,path,method,'paired',repeat,index,predfile,timingfile)
   sub=[r for r in timings if r['path']==path and r['N']==n and r['phase']=='paired']
   print(json.dumps(dict(cell=[path,n],summaries={method:summarize([r for r in sub if r['method']==method]) for method in METHODS})),flush=True)
  for index,group in enumerate(fresh):
   order=METHODS if index%2==0 else list(reversed(METHODS))
   for method in order:request(group,32,'prefix_share_batch',method,'fresh_confirmation',0,index,predfile,timingfile)
 for method,m in models.items():assert versions[method]==tuple(p._version for p in m.parameters()) and not any(p.requires_grad or p.grad is not None for p in m.parameters())
 summaries=[]
 for path,n in CELLS:
  for method in METHODS:
   sub=[r for r in timings if r['phase']=='paired' and r['path']==path and r['N']==n and r['method']==method]
   summaries.append(dict(path=path,N=n,method=method,**summarize(sub)))
 quality=[]
 for phase in ['paired','fresh_confirmation']:
  by={method:{r['item_id']:r for r in records if r['phase']==phase and r['path']=='prefix_share_batch' and r['N']==32 and r['repeat']==0 and r['method']==method} for method in METHODS}
  baseline=by['unmerged'];candidate=by['merged'];assert baseline.keys()==candidate.keys()
  probabilities=[]
  for k in baseline:
   za=torch.tensor(baseline[k]['logits']);zb=torch.tensor(candidate[k]['logits']);probabilities.append(float((za.softmax(-1)-zb.softmax(-1)).abs().max()))
  quality.append(dict(phase=phase,count=len(baseline),baseline_correct=sum(r['correct'] for r in baseline.values()),merged_correct=sum(r['correct'] for r in candidate.values()),flips=sum(baseline[k]['prediction']!=candidate[k]['prediction'] for k in baseline),gains=sum(not baseline[k]['correct'] and candidate[k]['correct'] for k in baseline),losses=sum(baseline[k]['correct'] and not candidate[k]['correct'] for k in baseline),max_probability_diff=max(probabilities),mean_max_probability_diff=statistics.mean(probabilities)))
 dump(a.output/'report.json',dict(status='BOUNDED_LORA_MERGE_COMPARISON',workflow='FastForward',model=str(a.model),adapter=str(a.adapter),adapter_sha256=sha(a.adapter/'adapter_model.safetensors'),dataset=str(a.dataset),dataset_sha256={n:sha(a.dataset/n) for n in ['inference.json','gold.json','merge_confirmation_inference.json','merge_confirmation_gold.json']},summaries=summaries,quality=quality,pid=evidence['pid'],gpu_uuid=evidence['gpu_uuid'],environment={p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},training=False,checkpoint_written=False,merge='FP32 safe merge then BF16 storage,252 targets checked',timing='Same paper complete-readout boundary: decoded image + extracted questions through preprocess/tokenization,H2D,full shared path and original LM readout; synchronized; no disk/decode/network/queue/model load/merge/CPU scoring',paired='Two versions co-resident on same exclusive H100,one PID,alternating per image/request;3 repeats of same5 measured groups after2 distinct warmup groups',limitations=['Existing development timing fixture; 3 repeats are not more unique questions.','Eight additional disjoint GQA-val images/256questions are bounded confirmation,not full benchmark.','Merging is algebraically equivalent in real arithmetic; BF16 weight rounding can change probabilities and answers.','Two resident model copies affect memory/cache conditions; relative paired timing only,not standalone deployment peak memory.','Only optimization is weight merge; no visual compression,layer truncation,training,token/prompt/readout changes.']))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['model','adapter','dataset','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 try:main(a)
 except Exception as e:invalidate(a.output,e);raise
