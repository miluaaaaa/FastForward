"""Same merged weights; profile full-model versus targeted FP32 LoRA merge."""
import argparse,time
from pathlib import Path
import torch
from peft import PeftModel
from safetensors.torch import load_file
from shunong_accel._legacy.vdm.models.vdm_model import VDM
from transformers import AutoProcessor,AutoConfig,Qwen3VLForConditionalGeneration
from shunong_accel._legacy.eval_visual_jev_merge import exact_adapter
from shunong_accel._legacy.visual_jev_compat import install_metadata_shim
from shunong_accel._legacy.visual_jev_paper_compat import install as install_paper_compat
from shunong_accel._legacy.fundm_sweep_contract import SweepHead

def load_profile(c,mode):
 profile={};total=time.perf_counter()
 def stage(name,fn):
  torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();profile[name]=time.perf_counter()-t;return v
 def base():
  m=VDM.__new__(VDM);torch.nn.Module.__init__(m);m.model_path=c['model'];m.device_str='cuda'
  m.processor=stage('processor',lambda:AutoProcessor.from_pretrained(c['model']));m.tokenizer=m.processor.tokenizer;m.config=stage('config',lambda:AutoConfig.from_pretrained(c['model']))
  kwargs=dict(dtype=torch.bfloat16,attn_implementation='sdpa')
  if mode=='direct_load':kwargs['device_map']={'':'cuda'}
  m.backbone=stage('weight_from_pretrained',lambda:Qwen3VLForConditionalGeneration.from_pretrained(c['model'],**kwargs));stage('to_cuda',lambda:m.backbone.to('cuda'));m.backbone.eval();m.heads=None;m.image_token_id=m.config.image_token_id;m.keep_full_lm_logits=False;m._option_ids=m._build_option_ids();return m
 m=stage('base_model_processor',base)
 m.processor.image_processor.max_pixels=200704;install_metadata_shim(m);install_paper_compat(m)
 m.backbone=stage('attach_adapter',lambda:PeftModel.from_pretrained(m.backbone,c['adapter'],local_files_only=True,is_trainable=False).eval().requires_grad_(False))
 def check():
  ck=load_file(str(Path(c['adapter'])/'adapter_model.safetensors'));exact_adapter(m,ck)
 stage('exact_adapter_check',check)
 if mode in ['original','direct_load']:
  buffers={n:b.detach().clone() for n,b in m.vl.named_buffers()};stage('whole_model_fp32',lambda:m.float());m.backbone=stage('safe_merge',lambda:m.backbone.merge_and_unload(safe_merge=True).eval().requires_grad_(False));stage('whole_model_bf16',lambda:m.bfloat16())
  for n,b in m.vl.named_buffers():b.data=buffers[n].to(b.device)
 else:
  assert mode=='targeted'
  def merge():
   count=0
   for module in list(m.backbone.modules()):
    if not hasattr(module,'lora_A') or 'default' not in module.lora_A:continue
    assert not module.fan_in_fan_out and not module.use_dora.get('default',False)
    weight=module.get_base_layer().weight;assert weight.dtype==torch.bfloat16
    delta=module.get_delta_weight('default');assert delta.dtype==torch.float32
    merged=weight.float()+delta;assert torch.isfinite(merged).all();weight.copy_(merged.to(weight.dtype));count+=1
   assert count==252
   return m.backbone.unload().eval().requires_grad_(False)
  m.backbone=stage('targeted_fp32_merge_unload',merge)
 def headload():
  h=SweepHead(m.config.text_config.hidden_size,'linear').cuda();h.load_state_dict(torch.load(c['head'],map_location='cpu',weights_only=False)['state_dict']);return h.eval().requires_grad_(False)
 head=stage('head',headload);assert len(m.vl_model.language_model.layers)==36 and not any('lora_' in n for n,_ in m.named_parameters());profile['total_seconds']=time.perf_counter()-total;return m,head,profile

def exact_models(a,b):
 counts={}
 for label,one,two in [('parameters',dict(a.named_parameters()),dict(b.named_parameters())),('buffers',dict(a.named_buffers()),dict(b.named_buffers()))]:
  assert one.keys()==two.keys()
  for name,x in one.items():
   y=two[name];assert x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y),(label,name)
  counts[label]=len(one)
 return counts
