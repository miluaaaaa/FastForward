"""Restore standard causal attention under Transformers4.57 Qwen3-VL.

Its no-mask/no-cache branch mistakes repeated M-RoPE temporal coordinates for
packed text boundaries. Explicit all-valid masks prevent that detection. This
is a compatibility correction, not a model optimization. Author files unchanged.
"""
from types import SimpleNamespace
import torch

def causal_mask_hook(module,args,kwargs):
 if kwargs.get('attention_mask') is None and kwargs.get('past_key_values') is None:
  inputs=kwargs.get('inputs_embeds')
  if inputs is None:inputs=kwargs.get('input_ids')
  if inputs is None:raise ValueError('Cannot bind valid-token mask without inputs')
  kwargs=dict(kwargs)
  kwargs['attention_mask']=torch.ones(inputs.shape[:2],device=inputs.device,dtype=torch.long)
 return args,kwargs

def install(m):
 original=m.vl_model.get_image_features
 def features(*args,return_dict=None,**kwargs):
  result=original(*args,**kwargs)
  return SimpleNamespace(pooler_output=result[0],deepstack_features=result[1]) if return_dict else result
 m.vl_model.get_image_features=features
 m.vl_model.language_model.register_forward_pre_hook(causal_mask_hook,with_kwargs=True)
 return dict(image_features='Only explicit return_dict=True translates original tuple to named fields',attention='Explicit all-valid 2D mask for no-mask/no-cache inputs avoids false packed-text detection from repeated M-RoPE temporal positions; existing padded masks and cached paths preserved')
