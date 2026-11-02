"""Continuation with explicit approximate token updates and private low-rank weights."""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from fastforward_accel._legacy.broad_runtime import BroadJoint
from fastforward_accel._legacy.broad_kernels import qk_pair
from fastforward_accel._legacy.lower_kernels import norm_rope
from fastforward_accel._legacy.search_kernels import swiglu
from fastforward_accel._legacy.limit_kernels import attention_direct, lm_readout, head_readout, gateup_swiglu, StaticFP8Linear
from fastforward_accel._legacy.sub10_runtime import text_parts


class CalibrationLinear:
    def __init__(self,weight):self.weight=weight;self.maximum=0.
    def __call__(self,x):
        self.maximum=max(self.maximum,float(x.float().abs().amax()))
        return F.linear(x.reshape(-1,x.shape[-1]),self.weight)


class LimitJoint(BroadJoint):
    def configure(self,spec,prototype):
        super().configure(spec,prototype)
        f=self.features;lm=self.m.vl_model.language_model
        count=f.get('partial_ffn_layers',0)
        self.partial_mlp={id(lm.layers[j].mlp) for j in self.language_indices[-count:]} if count else set()
        count=f.get('frozen_context_layers',0)
        self.partial_attention={id(lm.layers[j].self_attn) for j in self.language_indices[-count:]} if count else set()
        if self.partial_attention:
            self.partial_mlp|={id(lm.layers[j].mlp) for j in self.language_indices[-count:]}
            self.context_kv={id(lm.layers[j].self_attn):torch.cat((lm.layers[j].self_attn.k_proj.weight,
                lm.layers[j].self_attn.v_proj.weight)).contiguous() for j in self.language_indices[-count:]}
        self.factorized={};ranks=[]
        self.static_attention={};self.static_mlp={}
        if f.get('static_fp8'):
            def operator(w):
                return CalibrationLinear(w) if f.get('fp8_calibrate') else StaticFP8Linear(w,f.get('fp8_inverse_scale',16.))
            for j in self.language_indices:
                layer=lm.layers[j];qw,gw=self.packed[j]
                if f['static_fp8'] in ('attention','all'):
                    self.static_attention[id(layer.self_attn)]=(operator(qw),operator(layer.self_attn.o_proj.weight))
                if f['static_fp8'] in ('mlp','all'):
                    gate,down=self.narrow.get(id(layer.mlp),(gw,layer.mlp.down_proj.weight))
                    self.static_mlp[id(layer.mlp)]=(operator(gate),operator(down))
            self.spec['numeric_format']='FP8 E4M3FN static activation/tensor weight scale, fast accumulation: '+f['static_fp8']
        count=f.get('lowrank_layers',0)
        if count:
            assert f['lowrank_target'] in ('down','gate','both')
            # A fixed, setup-only, randomized SVD. No labels, activation fitting or checkpoint mutation.
            for j in self.language_indices[-count:]:
                layer=lm.layers[j];packed,down=self.narrow.get(id(layer.mlp),(self.packed[j][1],layer.mlp.down_proj.weight))
                factors=[]
                for name,w in [('gate',packed),('down',down)]:
                    if f['lowrank_target'] not in (name,'both'):
                        factors.append(w);continue
                    rank=max(128,int(min(w.shape)*f['lowrank_ratio'])//128*128)
                    rank=min(rank,min(w.shape));original=w.float()
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(19031+j)
                        u,s,v=torch.svd_lowrank(original,q=rank,niter=f.get('lowrank_iterations',1))
                    left=(u*s).to(w.dtype).contiguous();right=v.T.to(w.dtype).contiguous()
                    energy=float(s.square().sum()/original.square().sum())
                    if f.get('lowrank_rescale'):left=left/(energy**.5)
                    factors.append((left,right))
                    ranks.append(dict(layer=j,target=name,rank=rank,retained_frobenius_energy=energy,
                                      original_shape=list(w.shape),rescale=bool(f.get('lowrank_rescale'))))
                self.factorized[id(layer.mlp)]=factors
        self.spec['limit_operators']={k:v for k,v in f.items() if k.startswith(('direct_attention','triangular',
            'fused_readout','partial_ffn','frozen_context','active_tail','lowrank','static_fp8','fp8_',
            'fused_gateup','gate_','prepare_buffer'))}
        self.spec['lowrank_matrices']=ranks
        if f.get('prepare_buffer'):
            capacity=max(self.pos)
            self.host_ids=np.empty(capacity,dtype=np.int64);self.host_ids[:len(self.prefix_array)]=self.prefix_array
            self.host_picks=np.zeros(9,dtype=np.int64)
            self.ids_tensor=torch.from_numpy(self.host_ids);self.picks_tensor=torch.from_numpy(self.host_picks)
        if f.get('fp8_calibrate'):
            rows=getattr(self,'calibration_rows',[prototype])
            assert len(rows)==f['fp8_calibrate']
            for row in rows:
                g=self.prepare(row,'candidate');b=next(b for b in self.pos if b>=len(g['ids']))
                self.core_joint(self.args(g,b));torch.cuda.synchronize()
            calibration=[]
            margin=f.get('fp8_calibration_margin',1.5)
            for kind,mapping in [('attention',self.static_attention),('mlp',self.static_mlp)]:
                for ident,operators in list(mapping.items()):
                    final=[]
                    for number,op in enumerate(operators):
                        maximum=max(op.maximum,1e-12)
                        inverse=2.**math.floor(math.log2(448./(maximum*margin)))
                        final.append(StaticFP8Linear(op.weight,inverse))
                        calibration.append(dict(kind=kind,slot=number,maximum=op.maximum,inverse_scale=inverse))
                    mapping[ident]=tuple(final)
            self.spec['static_fp8_calibration']=dict(item_ids=[row['item_id'] for row in rows],
                count=len(rows),margin=margin,supervised=False,operators=calibration)

    def prepare(self,row,backend):
        if not self.features.get('prepare_buffer'):return super().prepare(row,backend)
        import fastforward_accel._legacy.sub10_runtime as sub10_runtime
        start=time.perf_counter_ns();rgb=sub10_runtime.decode(row['image_path'],self.side,backend);decoded=time.perf_counter_ns()
        prefix,suffix=text_parts(row,self.style)
        encoding=self.m.tokenizer.backend_tokenizer.encode(suffix,add_special_tokens=False)
        tokens=encoding.ids;total=len(self.prefix_array)+len(tokens)
        if total>len(self.host_ids):raise RuntimeError('Request exceeds prepared serial buffer')
        self.host_ids[len(self.prefix_array):total]=tokens;self.host_picks.fill(0)
        cursor=suffix.index('\nOptions:\n')+len('\nOptions:\n')
        for i,opt in enumerate(row['candidates']):
            line=f'{chr(65+i)}. {opt}';assert suffix[cursor:cursor+len(line)]==line
            token=encoding.char_to_token(cursor+len(line.rstrip())-1);assert token is not None
            self.host_picks[i]=len(self.prefix_array)+token;cursor+=len(line)+1
        self.host_picks[8]=total-1
        return dict(rgb=rgb,ids=self.ids_tensor[:total],picks=self.picks_tensor,k=len(row['candidates']),
            prompt_text=prefix+suffix,decoded_ns=decoded,prepare_ns=time.perf_counter_ns(),start_ns=start)

    @staticmethod
    def projected(x,w):
        return F.linear(F.linear(x,w[1]),w[0]) if isinstance(w,tuple) else F.linear(x,w)

    def mlp(self,module,w,x):
        def full(z):
            if id(module) in self.static_mlp:
                gate,down=self.static_mlp[id(module)]
                g,u=gate(z).chunk(2,-1)
                value=down(swiglu(g,u,self.features.get('mlp_block',2048),self.features.get('mlp_warps',8)))
                return value.reshape(*z.shape[:-1],value.shape[-1])
            if self.features.get('fused_gateup'):
                assert id(module) not in self.factorized
                packed,down=self.narrow.get(id(module),(w,module.down_proj.weight))
                f=self.features
                value=gateup_swiglu(z,packed,f.get('gate_bm',32),f.get('gate_bn',64),f.get('gate_bk',64),f.get('gate_warps',4))
                return F.linear(value,down)
            if id(module) not in self.factorized:return super(LimitJoint,self).mlp(module,w,z)
            gate,down=self.factorized[id(module)]
            g,u=self.projected(z,gate).chunk(2,-1)
            return self.projected(swiglu(g,u,self.features.get('mlp_block',2048),self.features.get('mlp_warps',8)),down)
        if id(module) not in self.partial_mlp or x.ndim!=3:return full(x)
        selected=self.active_positions
        value=full(x[0].index_select(0,selected))
        result=torch.zeros_like(x);result[0].index_copy_(0,selected,value)
        return result

    def attention(self,a,w,hidden_states,position_embeddings,attention_mask,**kwargs):
        if id(a) in self.partial_attention:
            t=hidden_states.shape[1];d=a.head_dim;selected=self.active_positions
            narrow=hidden_states[0].index_select(0,selected)
            q=F.linear(narrow,a.q_proj.weight).view(1,-1,a.q_proj.out_features//d,d)
            k,v=F.linear(hidden_states,self.context_kv[id(a)]).chunk(2,-1)
            k=k.view(1,t,-1,d);v=v.view(1,t,-1,d).transpose(1,2)
            cos,sin=position_embeddings
            q=norm_rope(q.contiguous(),a.q_norm,cos.index_select(1,selected),sin.index_select(1,selected))
            k=norm_rope(k.contiguous(),a.k_norm,cos,sin)
            out=attention_direct(q,k,v,a.scaling,self.features.get('attn_bm',16),self.features.get('attn_bn',64),
                self.features.get('attn_warps',4),selected)
            value=a.o_proj(out.reshape(len(selected),-1))
            result=torch.zeros_like(hidden_states);result[0].index_copy_(0,selected,value)
            return result,None
        if not self.features.get('direct_attention') and id(a) not in self.static_attention:return super().attention(a,w,hidden_states,position_embeddings,attention_mask,**kwargs)
        assert attention_mask is None
        shape=hidden_states.shape[:-1];d=a.head_dim
        if id(a) in self.static_attention:
            projection,output=self.static_attention[id(a)]
            packed=projection(hidden_states).view(*shape,-1)
        else:packed=F.linear(hidden_states,w)
        q,k,v=packed.split([a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features],-1)
        q,k=qk_pair(q.view(*shape,-1,d),k.view(*shape,-1,d),a.q_norm,a.k_norm,*position_embeddings,
            self.features.get('qk_rows',4),self.features.get('qk_warps',4),self.features.get('qk_half',False))
        v=v.view(*shape,-1,d).transpose(1,2)
        out=attention_direct(q,k,v,a.scaling,self.features.get('attn_bm',16),self.features.get('attn_bn',64),
            self.features.get('attn_warps',4),triangular=self.features.get('triangular',True))
        flattened=out.reshape(*shape,-1)
        if id(a) in self.static_attention:return output(flattened).view(*shape,-1),None
        return a.o_proj(flattened),None

    def core_joint(self,args,native=False):
        if native:return super().core_joint(args,True)
        rgb,ids,picks,mask=args
        # Positions depend on the fresh request, including its true (unpadded) end.
        if self.partial_mlp or self.partial_attention:
            tail=self.features.get('active_tail',16)
            tail_positions=(picks[8]-torch.arange(tail-1,-1,-1,device=picks.device)).clamp_min(0)
            self.active_positions=torch.cat((picks,tail_positions)).contiguous()
        if not self.features.get('fused_readout'):return super().core_joint(args)
        lm=self.m.vl_model.language_model
        visual,deep=self.vision(rgb);h=self.m.vl_model.get_input_embeddings()(ids)
        h[0].index_copy_(0,self.vi,visual);pos=self.pos[ids.shape[1]];rope=lm.rotary_emb(h,pos)
        normalized=self.norm(h,lm.layers[self.language_indices[0]].input_layernorm);chosen=None
        for offset,j in enumerate(self.language_indices):
            layer=lm.layers[j];qw,gw=self.packed[j];final=offset==len(self.language_indices)-1
            injection=deep[j] if j<len(deep) else None
            if final and self.features.get('terminal_queries') and injection is None:
                chosen=self.terminal(layer,qw,gw,h,normalized,rope,picks);break
            attn,_=self.attention(layer.self_attn,qw,hidden_states=normalized,position_embeddings=rope,attention_mask=None)
            residual,post=self.post_attention(h,attn,layer)
            if final and self.selective_tail and injection is None:
                selected=picks if self.readout=='head' else picks[8:9]
                chosen=residual[0].index_select(0,selected)+self.mlp(layer.mlp,gw,post[0].index_select(0,selected))
            else:
                value=self.mlp(layer.mlp,gw,post)
                if final:h=self.inject_reference(residual+value,injection)
                else:h,normalized=self.carry(residual,value,injection,lm.layers[self.language_indices[offset+1]].input_layernorm,self.image_start,self.Nv)
        if chosen is None:chosen=h[0].index_select(0,picks)
        if self.readout=='head':return head_readout(chosen.contiguous(),self.heads[str(self.depth)],mask)
        last=chosen if self.selective_tail else chosen[8][None]
        return lm_readout(last.contiguous(),lm.norm,self.option_weight)
