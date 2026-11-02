"""Fixed-size first-image path: fresh decode, fused raw-pixel packing, joint vision/LM graph."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torchvision.io import read_file,decode_image,ImageReadMode
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
from fastforward_accel._legacy.vdm.models.prompts import prefix_text, suffix_text
from fastforward_accel._legacy.vision_graph_runtime import VisionGraph
from fastforward_accel._legacy.first_response_runtime import FirstResponse36
from fastforward_accel._legacy.fundm_sweep_contract import SweepHead
from fastforward_accel._legacy.serving_dynamic_norm import fused_norm

CAPS=list(range(64,257,32))+[320,384,512,768,1024]
@triton.jit
def pack_rgb(X,Y,S:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N;p=i//1536;d=i%1536
    c=d//512;y0=(d%256)//16;x0=d%16;gw=S//16
    gy=p//(gw//2*4);gx=(p//4)%(gw//2);my=(p%4)//2;mx=p%2
    y=gy*32+my*16+y0;x=gx*32+mx*16+x0
    raw=tl.load(X+(y*S+x)*3+c,valid,other=0).to(tl.float32)
    # HF fast image processor fuses rescale into mean/std: (u8-127.5)/127.5.
    value=(raw-127.5)/127.5;tl.store(Y+i,value,valid)

@triton.jit
def rms_one(X,W,Y,D:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,B);x=tl.load(X+row*D+i,i<D,other=0).to(tl.float32)
    variance=tl.sum(x*x,0)/D;w=tl.load(W+i,i<D,other=0).to(tl.float32)
    normalized=(x*tl.rsqrt(variance+EPS)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+row*D+i,normalized*w,i<D)
def fast_norm(x,module):
    assert x.is_contiguous() and x.dtype==module.weight.dtype==torch.bfloat16
    out=torch.empty_like(x);d=x.shape[-1]
    rms_one[(x.numel()//d,)](x,module.weight,out,d,module.variance_epsilon,triton.next_power_of_2(d),num_warps=4 if d<=2048 else 8,enable_fp_fusion=False)
    return out

def text_parts(row,style):
    if style=='standard':return prefix_text(''),suffix_text(row['instruction'],row['candidates'],'choice')
    prefix='<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'
    options='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(row['candidates']))
    suffix='\n'+row['instruction'].strip()+'\nOptions:\n'+options+'\nReply with one letter.<|im_end|>\n<|im_start|>assistant\nAnswer:'
    if style=='minimal':suffix=suffix.replace('\nReply with one letter.','')
    return prefix,suffix

def decode(path,side,backend):
    if backend.startswith('cv'):
        flags=cv2.IMREAD_REDUCED_COLOR_4 if backend=='cvreduced' else cv2.IMREAD_COLOR
        image=cv2.imread(str(path),flags);assert image is not None
        inter=cv2.INTER_AREA if backend=='cvarea' else cv2.INTER_LINEAR
        return cv2.cvtColor(cv2.resize(image,(side,side),interpolation=inter),cv2.COLOR_BGR2RGB)
    if backend=='pilnearest':
        with Image.open(path) as image:
            image.draft('RGB',(side,side));return np.array(image.resize((side,side),Image.Resampling.NEAREST).convert('RGB'),copy=True)
    if backend=='tv':
        rgb=decode_image(read_file(str(path)),mode=ImageReadMode.RGB).permute(1,2,0).contiguous().numpy();image=Image.fromarray(rgb)
        return np.array(image.resize((side,side),Image.Resampling.BILINEAR),copy=True)
    with Image.open(path) as image:
        image.draft('RGB',(side,side));resized=image.resize((side,side),Image.Resampling.BILINEAR).convert('RGB')
        return np.array(resized,copy=True)

class Joint(FirstResponse36):
    def __init__(self,m,head,packed,heads):
        super().__init__(m,head,packed,'implicit',CAPS);self.heads=heads;self.vg=VisionGraph(m);self.option_weight=m.lm_head.weight.index_select(0,m.option_token_ids).contiguous();self.cache={};self.builds=[]
    def attention(self,a,w,*args,**kwargs):
        if self.spec.get('rotary')=='fused':
            from fastforward_accel._legacy.rotary_exact_runtime import attention
            return attention(a,w,*args,**kwargs)
        return super().attention(a,w,*args,**kwargs)
    def configure(self,spec,prototype):
        self.cache.clear();self.builds=[];self.spec=spec;self.norm=fast_norm if spec.get('norm')=='fast' else fused_norm;
        import fastforward_accel._legacy.packing_layout_runtime as packing_layout_runtime
        packing_layout_runtime.fused_norm=self.norm
        self.side=spec['side'];self.depth=spec['depth'];self.readout=spec.get('readout','lm' if self.depth==len(self.m.vl_model.language_model.layers) else 'head');self.style=spec['prompt'];self.language_indices=spec.get('language_indices',list(range(self.depth)));self.vision_indices=spec.get('vision_indices',list(range(24)));self.vg.active_indices=set(self.vision_indices);self.Nv=(self.side//32)**2
        prefix,_=text_parts(prototype,self.style);tokens=self.m.tokenizer(prefix,add_special_tokens=False)['input_ids'];image=self.m.image_token_id;assert tokens.count(image)==1
        at=tokens.index(image);self.prefix=torch.tensor(tokens[:at]+[image]*self.Nv+tokens[at+1:],dtype=torch.long);self.vi=torch.arange(at,at+self.Nv,device='cuda')
        self.grid_cpu=torch.tensor([[1,self.side//16,self.side//16]],dtype=torch.long);self.grid=self.grid_cpu.cuda();self.geo=self.vg.geo(self.grid)
        self.pos={}
        for b in CAPS:
            if b<len(self.prefix)+1:continue
            ids=torch.cat((self.prefix,torch.zeros(b-len(self.prefix),dtype=torch.long)))[None]
            pos,_=self.m.vl_model.get_rope_index(ids,mm_token_type_ids=(ids==image).int(),image_grid_thw=self.grid_cpu)
            self.pos[b]=pos.cuda()
        style=spec.get('answer_tokens','space');strings=[(' ' if style=='space' else '')+chr(65+i) for i in range(8)]
        labels=[self.m.tokenizer(t,add_special_tokens=False)['input_ids'] for t in strings];assert all(len(x)==1 for x in labels)
        self.option_weight=self.m.lm_head.weight.index_select(0,torch.tensor([x[0] for x in labels],device='cuda')).contiguous()
        self.versions=tuple(p._version for p in self.m.parameters())
    def prepare(self,row,backend):
        start=time.perf_counter_ns();rgb=decode(row['image_path'],self.side,backend);decoded=time.perf_counter_ns()
        prefix,suffix=text_parts(row,self.style);enc=self.m.tokenizer(suffix,add_special_tokens=False,return_offsets_mapping=True)
        full=torch.cat((self.prefix,torch.tensor(enc['input_ids'],dtype=torch.long)));cursor=suffix.index('\nOptions:\n')+len('\nOptions:\n');ends=[]
        for i,opt in enumerate(row['candidates']):
            line=f'{chr(65+i)}. {opt}';assert suffix[cursor:cursor+len(line)]==line
            last=cursor+len(line.rstrip())-1;hits=[j for j,(a,b) in enumerate(enc['offset_mapping']) if a<=last<b];assert len(hits)==1
            ends.append(len(self.prefix)+hits[0]);cursor+=len(line)+1
        picks=torch.zeros(9,dtype=torch.long);picks[:len(ends)]=torch.tensor(ends);picks[8]=len(full)-1
        return dict(rgb=rgb,ids=full,picks=picks,k=len(ends),prompt_text=prefix+suffix,decoded_ns=decoded,prepare_ns=time.perf_counter_ns(),start_ns=start)
    def args(self,g,b):
        ids=torch.zeros((1,b),dtype=torch.long);ids[0,:len(g['ids'])].copy_(g['ids'])
        return torch.from_numpy(g['rgb']).cuda(),ids.cuda(),g['picks'].cuda(),(torch.arange(8)<g['k'])[None].cuda()
    def vision(self,rgb,native=False):
        pixels=torch.empty(((self.side//16)**2,1536),device='cuda',dtype=self.m.dtype)
        pack_rgb[(triton.cdiv(pixels.numel(),1024),)](rgb,pixels,self.side,pixels.numel(),1024,enable_fp_fusion=False)
        if native:
            skipped=[]
            try:
                for j,block in enumerate(self.vg.v.blocks):
                    if j not in self.vg.active_indices:
                        old=block.__dict__.get('forward');skipped.append((block,old));block.forward=lambda hidden_states,*a,**kw:hidden_states
                image,deep=self.vg.original(pixels,self.grid);return torch.cat(image),deep
            finally:
                for block,old in skipped:
                    if old is None:block.__dict__.pop('forward',None)
                    else:block.forward=old
        return self.vg.core(pixels,self.geo)
    def core_joint(self,args,native=False):
        rgb,ids,picks,mask=args;b=ids.shape[1];lm=self.m.vl_model.language_model
        visual,deep=self.vision(rgb,native);h=self.m.vl_model.get_input_embeddings()(ids);h[0].index_copy_(0,self.vi,visual)
        pos=self.pos[b];rope=lm.rotary_emb(h,pos);cache=torch.arange(b,device='cuda')
        for j in self.language_indices:
            layer=lm.layers[j]
            if native:h=layer(h,position_embeddings=rope,attention_mask=None,position_ids=pos[0],cache_position=cache,use_cache=False)
            else:
                qw,gw=self.packed[j];attn,_=self.attention(layer.self_attn,qw,hidden_states=self.norm(h,layer.input_layernorm),position_embeddings=rope,attention_mask=None,position_ids=pos[0],cache_position=cache,use_cache=False)
                h=h+attn;h=h+self.mlp(layer.mlp,gw,self.norm(h,layer.post_attention_layernorm))
            if j<len(deep):
                add=torch.zeros_like(h);add[0].index_copy_(0,self.vi,deep[j]);h=h+add
        chosen=h[0].index_select(0,picks)
        if self.readout=='head':return self.heads[str(self.depth)].score(chosen[8][None],chosen[:8][None],mask)
        return F.linear(lm.norm(chosen[8][None]).to(self.m.dtype),self.option_weight).float()
    def prebuild(self,g):
        start=time.perf_counter();limit=len(g['ids'])
        for b in self.pos:
            clipped=dict(g);clipped['ids']=g['ids'][:b];clipped['picks']=g['picks'].clamp_max(min(limit,b)-1)
            static=self.args(clipped,b);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):self.core_joint(static)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=self.core_joint(static)
            self.cache[b]=(static,graph,out)
        torch.cuda.synchronize();self.builds=list(self.cache);return time.perf_counter()-start
    def run_joint(self,g):
        b=next((b for b in self.cache if b>=len(g['ids'])),None)
        if b is None:
            b=len(g['ids']);ids=g['ids'][None];pos,_=self.m.vl_model.get_rope_index(ids,mm_token_type_ids=(ids==self.m.image_token_id).int(),image_grid_thw=self.grid_cpu);self.pos[b]=pos.cuda()
            return self.core_joint(self.args(g,b)),False
        static,graph,out=self.cache[b]
        # Same three request inputs always refreshed; no feature/answer reuse.
        for dest,src in zip(static,[torch.from_numpy(g['rgb']),F.pad(g['ids'],(0,b-len(g['ids'])))[None],g['picks'],(torch.arange(8)<g['k'])[None]]):dest.copy_(src)
        graph.replay();return out,True
    def gate(self,row,backend):
        g=self.prepare(row,backend);b=next(b for b in self.cache if b>=len(g['ids']));args=self.args(g,b)
        pos,_=self.m.vl_model.get_rope_index(g['ids'][None],mm_token_type_ids=(g['ids'][None]==self.m.image_token_id).int(),image_grid_thw=self.grid_cpu)
        assert torch.equal(pos,self.pos[b][:,:,:len(g['ids'])].cpu()),'Position template differs from actual prompt'
        pixels=torch.empty(((self.side//16)**2,1536),device='cuda',dtype=self.m.dtype);pack_rgb[(triton.cdiv(pixels.numel(),1024),)](args[0],pixels,self.side,pixels.numel(),1024,enable_fp_fusion=False)
        # Independently use official HF pixel normalization+patch reordering, no resize.
        reference=self.m.processor.image_processor.preprocess(images=[Image.fromarray(g['rgb'])],do_resize=False,return_tensors='pt')['pixel_values'].to('cuda',self.m.dtype)
        assert torch.equal(pixels,reference),('pixel_packing',float((pixels-reference).abs().max()))
        k=g['k'];native=self.core_joint(args,True)[0,:k].float().cpu();actual=self.run_joint(g)[0][0,:k].float().cpu()
        assert native.argmax()==actual.argmax() and torch.all((native-actual).abs()<=.003+.0001*native.abs()),('native_graph',native.tolist(),actual.tolist())
        changed=list(args);changed[1]=changed[1].clone();changed[1][:,len(g['ids']):]=42
        perturbed=self.core_joint(tuple(changed))[0,:k].float().cpu();assert torch.equal(perturbed,actual),'Padding tokens leak into real answer'
        return dict(passed=True,pixel_pack_exact=True,positions_exact=True,prompt_binding=True,padding_isolated=True,index=row['item_id'],native=native.tolist(),actual=actual.tolist())
