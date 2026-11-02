"""Extend exact pointwise RMSNorm fusion to native attention Q/K norms."""
from fastforward_accel._legacy.fused_norm_runtime import FusedGraph24, fused_norm

class QKFusedGraph24(FusedGraph24):
 def core(self,args):
  previous=[]
  try:
   for layer in self.layers:
    for norm in (layer.self_attn.q_norm,layer.self_attn.k_norm):
     previous.append((norm,'forward' in norm.__dict__,norm.__dict__.get('forward')))
     norm.forward=lambda x,module=norm:fused_norm(x,module)
   return super().core(args)
  finally:
   for norm,exists,value in previous:
    if exists:norm.forward=value
    else:norm.__dict__.pop('forward',None)
