"""Single-request staging: one aligned pinned host/device packet per graph."""
import torch


class Packet:
    def __init__(self,shapes,dtypes):
        sizes=[torch.empty((),dtype=d).element_size()*int(torch.tensor(s).prod()) for s,d in zip(shapes,dtypes)]
        offsets=[];total=0
        for n in sizes:
            total=(total+255)//256*256;offsets.append(total);total+=n
        self.host=torch.empty(total,dtype=torch.uint8,pin_memory=True)
        self.device=torch.empty(total,dtype=torch.uint8,device='cuda')
        self.host.zero_()
        self.host_views=tuple(self.host[o:o+n].view(d).view(s) for o,n,d,s in zip(offsets,sizes,dtypes,shapes))
        self.device_views=tuple(self.device[o:o+n].view(d).view(s) for o,n,d,s in zip(offsets,sizes,dtypes,shapes))
        self.done=torch.cuda.Event();self.pending=False

    def fill(self,g):
        if self.pending and not self.done.query():self.done.synchronize()
        rgb,ids,picks,mask=self.host_views
        rgb.copy_(torch.from_numpy(g['rgb']))
        ids.zero_();ids[0,:len(g['ids'])].copy_(g['ids'])
        picks.copy_(g['picks']);mask.zero_();mask[0,:g['k']]=True
        self.device.copy_(self.host,non_blocking=True)
        self.done.record();self.pending=True
