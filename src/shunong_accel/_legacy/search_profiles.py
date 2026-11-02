"""100 preregistered structural configurations, plus an unchanged incumbent."""
import copy
import json
from collections import Counter

DEPTHS=[35,34,32,30,28,26,24,22,20,16]
RATIOS=[.95,.9,.875,.8,.75,.65,.6,.5,.375,.25]
COUNTS=[1,2,4,6,8,10,12,16,20,24]


def keep(total,count,mode,protected):
    protected=set(protected)
    assert len(protected)<=count<=total
    rest=[i for i in range(total) if i not in protected];n=count-len(protected)
    if mode=='early':chosen=rest[:n]
    else:
        assert mode=='uniform'
        chosen=[] if n==0 else [rest[len(rest)//2]] if n==1 else [rest[i*(len(rest)-1)//(n-1)] for i in range(n)]
    result=sorted(protected|set(chosen));assert len(result)==count
    return result


def profiles(config):
    result=[];parent=config['architecture_parent']
    control=copy.deepcopy(parent);control.pop('parent_id',None)
    control.update(id='current_full36',family='control');result.append(control)
    def add(family,**structure):
        p=copy.deepcopy(parent);p.pop('parent_id',None)
        number=1+sum(x['family']==family for x in result)
        p.update(id=f'{family}_{number:02}',family=family)
        structure.setdefault('depth',36);structure.setdefault('layer_mode','early')
        pairs=structure.get('fold_pairs',0)
        indices=keep(36,structure['depth'],structure['layer_mode'],[0,1,2,35])
        if pairs:
            assert structure['depth']==36
            structure['pairs']=[[33-2*i,34-2*i] for i in range(pairs)]
            removed={b for a,b in structure['pairs']};indices=[i for i in indices if i not in removed]
        p['features'].update(language_count=len(indices),language_mode='prefix',structural=structure)
        p['spec'].update(depth=len(indices),language_indices=indices)
        result.append(p)
    for depth in DEPTHS:add('depth_early',depth=depth)
    for depth in DEPTHS:add('depth_uniform',depth=depth,layer_mode='uniform')
    for n in COUNTS:add('attention_only_tail',skip_ffn=n)
    for n in COUNTS:add('ffn_only_tail',skip_attention=n)
    for ratio in RATIOS:add('ffn_width_all',ffn_ratio=ratio,ffn_scope='all')
    for ratio in RATIOS:add('ffn_width_tail',ffn_ratio=ratio,ffn_scope='last18')
    for n in [23,22,20,18,16,14,12,10,8,4]:add('vision_depth',vision_count=n)
    for scope,n in [('all',24),('all',16),('all',8),('all',4),('all',2),('all',1),
                    ('last18',24),('last18',16),('last18',8),('last18',4)]:
        add('attention_width',heads=n,head_scope=scope)
    for n in [1,2,4,6,8]:
        for gain in [1.,2.]:add('block_folding',fold_pairs=n,fold_gain=gain)
    hybrids=[dict(depth=32,ffn_ratio=.875,vision_count=20),
        dict(depth=28,ffn_ratio=.75,vision_count=16),dict(depth=24,ffn_ratio=.75,vision_count=12),
        dict(depth=20,ffn_ratio=.5,vision_count=8),dict(depth=16,ffn_ratio=.5,vision_count=4),
        dict(depth=32,skip_ffn=4,heads=16,head_scope='last18'),
        dict(depth=28,skip_attention=4,vision_count=12),dict(depth=24,heads=8,ffn_ratio=.5),
        dict(fold_pairs=4,fold_gain=1.,vision_count=12,ffn_ratio=.75),
        dict(fold_pairs=8,fold_gain=2.,vision_count=8,ffn_ratio=.5)]
    for h in hybrids:add('hybrid',**h)
    assert len(result)==101
    assert Counter(p['family'] for p in result[1:])==dict.fromkeys([p['family'] for p in result[1:]],10)
    # Two mild layer schedules coincide at35/34; replace those with explicit
    # middle cuts so all100 configurations have different executed structures.
    for p,drop in [(result[11],[18]),(result[12],[17,18])]:
        p['spec']['language_indices']=[j for j in range(36) if j not in drop]
        p['features']['structural']['layer_mode']='middle'
    signatures=[json.dumps(dict(indices=p['spec']['language_indices'],structure=p['features'].get('structural',{})),sort_keys=True) for p in result]
    assert len(set(signatures))==len(result)
    return result


def select(summary,limit):
    def loss(cid):return max(s['loss_pp'] for s in summary[cid].values())
    def speed(cid):return min(s['native_mean_ms']/s['mean_ms'] for s in summary[cid].values())
    eligible=[cid for cid in summary if loss(cid)<=10 and speed(cid)>1]
    if eligible:return sorted(eligible,key=lambda c:(-speed(c),loss(c),c))[:limit]
    return sorted(summary,key=lambda c:(loss(c),-speed(c),c))[:limit]
