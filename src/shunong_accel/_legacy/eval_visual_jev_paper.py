"""Bounded same-image GQA reproduction of the paper's execution experiment."""
import argparse,hashlib,importlib.metadata,json,math,statistics,time
from pathlib import Path
import torch
from PIL import Image
from peft import PeftModel
from safetensors.torch import load_file
from shunong_accel._legacy.vdm.models.vdm_model import VDM
from shunong_accel._legacy.vdm.bench import sweep as author
from shunong_accel._legacy.visual_jev_compat import install_metadata_shim
from shunong_accel._legacy.visual_jev_paper_compat import install as install_paper_compat
from shunong_accel._legacy.gpu_guard import require_allocation, prove, invalidate

PATHS=['independent','independent_batch','vision_cache','vision_cache_batch','prefix_share','prefix_share_batch']
NS=[1,2,4,8,16,32]
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def qs(group,n):return [{k:q[k] for k in ['qtype','instruction','candidates']} for q in group['questions'][:n]]
def logits(m,g,path):
 m.vl_model.rope_deltas=None
 return getattr(m,'run_'+path)(g)['lm_option_logits']
def complete(m,image,questions,path):
 if path=='independent':
  return torch.cat([logits(m,m.prepare_group(image,[q]),path) for q in questions])
 return logits(m,m.prepare_group(image,questions),path)
def original_timer(m,image,questions,path):
 if path in ['prefix_share','prefix_share_batch']:
  return author.timed_prefix_share(m,image,questions,'',path.endswith('_batch'))
 return getattr(author,'timed_'+path)(m,image,questions,'')
def prompt_check(m,g,questions):
 prefix=('<|im_start|>system\nYou are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain.<|im_end|>\n<|im_start|>user\n<|vision_start|>'+'<|image_pad|>'*int((g.prefix_ids==m.image_token_id).sum())+'<|vision_end|>')
 for i,q in enumerate(questions):
  suffix='\n\nQuestion: '+q['instruction'].strip()+'\nOptions:\n'+'\n'.join(chr(65+j)+'. '+v for j,v in enumerate(q['candidates']))+'\nReply with the single letter of the correct option.<|im_end|>\n<|im_start|>assistant\nAnswer:'
  assert torch.cat([g.prefix_ids,g.suffix_ids[i]]).tolist()==m.tokenizer.encode(prefix+suffix,add_special_tokens=False)

def main(a):
 require_allocation();a.output.mkdir(parents=True,exist_ok=True)
 fixture=json.loads((a.dataset/'inference.json').read_text());gold=json.loads((a.dataset/'gold.json').read_text())
 groups=fixture['sentinel_groups']+fixture['timing_groups'];assert len(groups)==9
 assert len({g['image_id'] for g in groups})==9
 assert all(len(g['questions'])==32 and len({q['question_id'] for q in g['questions']})==32 for g in groups)
 for g in groups:
  assert sha(Path(g['image_path']))==g['image_sha256']
  assert all('label' not in q and 'answer' not in q for q in g['questions'])
  for q in g['questions']:
   y=gold[q['item_id']];assert q['candidates'][y['index']]==y['answer'] and y['image_id']==g['image_id']
 torch.manual_seed(0);torch.set_grad_enabled(False)
 m=VDM(str(a.model),device='cuda',dtype=torch.bfloat16,with_heads=False)
 m.processor.image_processor.max_pixels=200704
 shim=install_metadata_shim(m)
 paper_compat=install_paper_compat(m)
 m.backbone=PeftModel.from_pretrained(m.backbone,str(a.adapter),is_trainable=False,local_files_only=True).eval().requires_grad_(False)
 checkpoint=load_file(a.adapter/'adapter_model.safetensors');named=dict(m.backbone.named_parameters())
 def verify_weights():
  count=0
  for k,v in checkpoint.items():
   target=k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
   assert torch.equal(named[target].detach().cpu(),v.to(named[target].dtype)),k
   count+=v.numel()
  assert len(checkpoint)==504 and count==33030144
 verify_weights()
 first=fixture['sentinel_groups'][0]
 with Image.open(first['image_path']) as f:image=f.convert('RGB')
 evidence=prove(m,lambda:complete(m,image,qs(first,1),'independent'),a.output)
 # Separately computed native model logits, no cache/readout helper reuse.
 g=m.prepare_group(image,qs(first,2));prompt_check(m,g,qs(first,2));reference=logits(m,g,'independent')
 for i in range(2):
  full=torch.cat([g.prefix_ids,g.suffix_ids[i]])[None].to('cuda');m.vl_model.rope_deltas=None
  native=m.backbone(input_ids=full,pixel_values=g.pixel_values.to('cuda',m.dtype),image_grid_thw=g.image_grid_thw.to('cuda'),position_ids=g.position_ids[i][:,None].to('cuda'),use_cache=False,logits_to_keep=1).logits[0,-1]
  torch.testing.assert_close(reference[i],native.index_select(-1,m.option_token_ids).float(),rtol=0,atol=0)
 # Float32 semantic gate separates cache/mask mistakes from BF16 rounding.
 parameter_dtypes={name:p.dtype for name,p in m.named_parameters()}
 original_buffers={name:b.detach().clone() for name,b in m.named_buffers()}
 m.float();g=m.prepare_group(image,qs(first,2));reference=logits(m,g,'independent');fp32=[]
 for path in PATHS[1:]:
  actual=logits(m,g,path);torch.testing.assert_close(reference,actual,rtol=1e-4,atol=.003)
  fp32.append(dict(path=path,max_logit_diff=float((actual-reference).abs().max())))
 reversed_group=m.prepare_group(image,list(reversed(qs(first,2))))
 rev=logits(m,reversed_group,'prefix_share_batch').flip(0)
 torch.testing.assert_close(reference,rev,rtol=1e-4,atol=.003)
 print('FP32_CAUSAL_PARITY_PASSED',flush=True)
 m.bfloat16()
 for name,p in m.named_parameters():p.data=p.data.to(parameter_dtypes[name])
 for name,b in m.named_buffers():b.data=original_buffers[name].to(b.device)
 del original_buffers
 # PEFT stores LoRA in FP32: BF16 casting would lose released precision.
 # Restore exact checkpoint tensors after the separate FP32 semantic gate.
 for key,value in checkpoint.items():
  target=key.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
  named[target].copy_(value.to(device=named[target].device,dtype=named[target].dtype))
 verify_weights();versions=tuple(p._version for p in m.parameters());del checkpoint
 sentinel=[]
 for group in fixture['sentinel_groups']:
  with Image.open(group['image_path']) as f:img=f.convert('RGB')
  questions=qs(group,32);g=m.prepare_group(img,questions);prompt_check(m,g,questions)
  ref=logits(m,g,'independent')
  for path in PATHS[1:]:
   actual=logits(m,g,path);assert torch.isfinite(actual).all()
   if path=='vision_cache':torch.testing.assert_close(actual,ref,rtol=0,atol=0)
   for i,q in enumerate(questions):
    k=len(q['candidates']);za=ref[i,:k];zb=actual[i,:k]
    sentinel.append(dict(image_id=group['image_id'],item_id=group['questions'][i]['item_id'],path=path,max_logit_diff=float((za-zb).abs().max()),max_probability_diff=float((za.softmax(-1)-zb.softmax(-1)).abs().max()),flip=int(za.argmax()!=zb.argmax())))
 dump(a.output/'preflight.json',dict(passed=True,same_pid_gpu=evidence['passed'],all_adapter_tensors_exact=504,adapter_parameters=33030144,original_prompt_exact=True,native_peft_logits_exact=True,fp32_semantic_gate=fp32,fp32_question_permutation_passed=True,shim=shim,paper_compat=paper_compat,bf16_sentinel=sentinel))
 print('PAPER_PROTOCOL_PREFLIGHT_PASSED',flush=True)
 records=[];timings=[];token_counts={}
 with (a.output/'predictions.jsonl').open('w') as pred:
  for panel in ['complete_readout','author_instrumented']:
   for n in NS:
    for path in PATHS:
     torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[];spans=[]
     for index,group in enumerate(fixture['timing_groups']):
      with Image.open(group['image_path']) as f:img=f.convert('RGB')
      questions=qs(group,n)
      torch.cuda.synchronize();start=time.perf_counter()
      if panel=='complete_readout':result=complete(m,img,questions,path)
      else:timer=original_timer(m,img,questions,path)
      torch.cuda.synchronize();elapsed=time.perf_counter()-start
      if panel=='complete_readout':
       # Outside timed interval, as in the paper's path-return boundary.
       values=result.float().cpu().tolist()
       for i,q in enumerate(group['questions'][:n]):
        k=len(q['candidates']);z=values[i][:k];pick=max(range(k),key=z.__getitem__)
        rec=dict(N=n,path=path,group_index=index,image_id=group['image_id'],item_id=q['item_id'],question_id=q['question_id'],logits=z,prediction=pick,gold=gold[q['item_id']]['index'],correct=pick==gold[q['item_id']]['index'],warmup=index<2)
        pred.write(json.dumps(rec)+'\n');records.append(rec)
      if index>=2:
       times.append(elapsed)
       if panel=='author_instrumented':spans.append(dict(timer.spans))
     rec=dict(panel=panel,N=n,path=path,group_ms=[x*1000 for x in times],group_ms_mean=statistics.mean(times)*1000,amortized_ms_per_question=statistics.mean(times)*1000/n,questions_per_second=n/statistics.mean(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,warmups=2,measured_groups=5,phases_s=spans)
     timings.append(rec);print(json.dumps({k:rec[k] for k in ['panel','N','path','group_ms_mean','amortized_ms_per_question']}),flush=True)
 for group in groups:
  with Image.open(group['image_path']) as f:img=f.convert('RGB')
  g=m.prepare_group(img,qs(group,1));token_counts[group['image_id']]=int((g.prefix_ids==m.image_token_id).sum())
 assert versions==tuple(p._version for p in m.parameters()) and not any(p.requires_grad or p.grad is not None for p in m.parameters())
 accuracy=[]
 for path in PATHS:
  sub=[r for r in records if r['path']==path and r['N']==32 and not r['warmup']]
  accuracy.append(dict(path=path,count=len(sub),correct=sum(r['correct'] for r in sub),accuracy=sum(r['correct'] for r in sub)/len(sub)))
 dump(a.output/'report.json',dict(status='BOUNDED_PAPER_PROTOCOL_RETEST',workflow='shunongLLM-A71C',model=str(a.model),adapter=str(a.adapter),adapter_sha256=sha(a.adapter/'adapter_model.safetensors'),dataset=str(a.dataset),dataset_inference_sha256=sha(a.dataset/'inference.json'),dataset_gold_sha256=sha(a.dataset/'gold.json'),timings=timings,accuracy=accuracy,pid=evidence['pid'],gpu_uuid=evidence['gpu_uuid'],device=torch.cuda.get_device_name(0),environment={p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},visual_tokens=token_counts,training=False,merged=False,timing='Decoded RGB image and extracted question records -> processor/tokenization,H2D,model,path return; synchronized perf_counter; excludes decode,disk,network,queue,CPU score/probability conversion',panels=dict(complete_readout='All actual original run_* paths including LM readout; independent prepare+forward repeated per question',author_instrumented='Exact released sweep timed_* functions; shared-prefix and serial vision-cache timing omits readout; vision-cache-batch timing redundantly encodes vision; do not silently conflate with full-answer path'),limitations=['H100 with torch2.7/Transformers4.57/PEFT0.18, not paper RTX5090 torch2.14/Transformers5.17; explicit causal-mask compatibility correction prevents legacy M-RoPE false packed-sequence detection.','Reconstructed bounded GQA-val subset, not original paper execution-test items or all benchmark/seed quality replication.','Natural same-image questions and original author converter, native aspect ratios/max_pixels200704; no repeated question to fill groups.','Five measured image groups plus two distinct warmup groups, two additional disjoint sentinel groups; changing N uses nested subsets.','BF16 execution can change predictions; all measured differences retained.','Released seed0 LoRA stays enabled unmerged for all paths; no off/on LoRA comparison.']))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['model','adapter','dataset','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 try:main(a)
 except Exception as e:invalidate(a.output,e);raise
