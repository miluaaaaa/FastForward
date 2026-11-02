"""One resident full 4B plus a replaceable fast-path graph/weight bank."""
import gc
import io
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from fastforward_accel._legacy.full_fourb_runtime import FullFourB, decode
from fastforward_accel._legacy.packing_layout_runtime import pack_weights
from fastforward_accel._legacy.fundm_sweep_contract import SweepHead
from fastforward_accel._legacy.limit_runtime import LimitJoint
from fastforward_accel._legacy.decode_study import candidate as old_decode
import fastforward_accel._legacy.sub10_runtime as sub10_runtime


class Reference(FullFourB):
    def prebuild(self,prototypes):
        for p in prototypes['vision']:
            self.vision.build(p['pixels'].to('cuda',self.model.dtype),p['grid'].cuda())
        self.vision.mode='graph'
        for p in prototypes['language']:self.build(self.inputs(p),'full_graph')


class Fast:
    runtime_class=LimitJoint
    def __init__(self,reference):
        self.reference=reference;self.model=reference.model;self.config=reference.config
        self.heads=torch.nn.ModuleDict()
        for depth,path in self.config['heads'].items():
            h=SweepHead(self.model.config.text_config.hidden_size,'linear')
            h.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
            self.heads[depth]=h.cuda().eval().requires_grad_(False)
        self.packed=pack_weights(tuple(self.model.vl_model.language_model.layers))
        self.rt=None;self.last={};self.profile=None;self.scratch={}
        self.versions=tuple(p._version for p in self.heads.parameters())
        sub10_runtime.decode=self.decoder

    def decoder(self,path,side,backend):
        if self.profile['decode'].startswith('turbo_'):
            import cv2
            import imagecodecs
            data=Path(path).read_bytes()
            if data.startswith(b'\xff\xd8'):
                rgb=imagecodecs.jpeg8_decode(data,outcolorspace='RGB')
            elif data.startswith(b'\x89PNG\r\n\x1a\n') and len(data)>29 and data[24:26]==b'\x08\x02' and data[28]==0:
                from fastforward_accel._legacy.private_decode import rgb8
                rgb=rgb8(data,2,True)
            else:
                with Image.open(io.BytesIO(data)) as im:rgb=np.array(im.convert('RGB'))
            interpolation={'turbo_area':cv2.INTER_AREA,'turbo_cubic':cv2.INTER_CUBIC,
                           'turbo_linear':cv2.INTER_LINEAR}[self.profile['decode']]
            return cv2.resize(rgb,(side,side),interpolation=interpolation)
        if self.profile['decode'].startswith('hq_fast_'):
            data=Path(path).read_bytes()
            if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data)>29 and data[24:26]==b'\x08\x02' and data[28]==0:
                from fastforward_accel._legacy.private_decode import rgb8
                rgb=Image.fromarray(rgb8(data,2,True),'RGB')
            else:
                with Image.open(io.BytesIO(data)) as opened:
                    if self.profile['decode']=='hq_fast_draft' and opened.format=='JPEG':
                        opened.draft('RGB',(side,side))
                    rgb=opened.convert('RGB')
            return np.array(rgb.resize((side,side),Image.Resampling.BICUBIC),copy=True)
        if self.profile['decode']=='hq':
            # Original full RGB decode, then high quality resize. No JPEG draft,
            # pre-resized input, image cache or excluded client-side resize.
            rgb=decode(path)
            return np.array(rgb.resize((side,side),Image.Resampling.BICUBIC),copy=True)
        if side not in self.scratch:
            self.scratch[side]=(np.empty((side,side,3),np.uint8),np.empty((side,side,3),np.uint8))
        return old_decode(path,side,'pil_split',self.scratch[side])[0]

    def configure(self,profile,binding):
        if self.rt is not None:
            self.rt.cache.clear();self.rt.packets.clear();self.rt.transport=None;self.rt=None
        gc.collect();torch.cuda.empty_cache();self.profile=profile
        rt=self.runtime_class(self.model,None,self.packed,self.heads,features=dict(profile['features']))
        rt.calibration_rows=[binding[b]['development'][i] for i in range(4) for b in ['MMStar','AI2D']]
        prototype=binding['MMStar']['development'][0]
        rt.configure(profile['spec'],prototype)
        if rt.readout=='lm':
            # Native VDM has 16 option slots; this API accepts at most 8.
            expected=self.model.lm_head.weight.index_select(0,self.model.option_token_ids[:8])
            assert torch.equal(rt.option_weight,expected),'Original option token rows changed'
        begin=time.perf_counter();rt.prebuild(rt.prepare(prototype,'candidate'))
        self.rt=rt
        return dict(id=profile['id'],seconds=time.perf_counter()-begin,actual_spec=rt.spec,
                    language_indices=rt.language_indices,side=rt.side,decode=profile['decode'],
                    graph_buckets=list(rt.cache),calibration_has_labels=False)

    def predict(self,image,question,candidates,item_id='request',mode=None):
        row=dict(image_path=str(image),instruction=question,candidates=candidates,item_id=item_id)
        g=self.rt.prepare(row,'candidate');z,hit=self.rt.run_joint(g)
        values=z[0,:len(candidates)].float().cpu().tolist()
        assert hit and all(math.isfinite(x) for x in values)
        pred=values.index(max(values));exp=[math.exp(x-max(values)) for x in values]
        self.last=dict(tokens=len(g['ids']),visual_tokens=self.rt.Nv,language_graph=hit,
            vision_graph=hit,grid=self.rt.grid_cpu.tolist(),
            decode_ms=(g['decoded_ns']-g['start_ns'])/1e6,
            prepare_ms=(g['prepare_ns']-g['decoded_ns'])/1e6)
        return dict(id=item_id,answer=dict(letter=chr(65+pred),text=candidates[pred]),
                    logits=values,probabilities=[x/sum(exp) for x in exp])

    def assert_frozen(self):
        self.reference.assert_frozen()
        assert self.versions==tuple(p._version for p in self.heads.parameters())
