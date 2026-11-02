"""Serial-request transport adapter; graph/model arithmetic stays unchanged.

One instance belongs to one runtime and CPU request thread. CUDA events protect
pinned source buffers from reuse before DMA finishes. This is not concurrent
serving: graph inputs and output buffers also belong to that serial runtime.
"""
import torch


class PinnedTransport:
    def __init__(self, runtime):
        self.runtime = runtime
        self.buffers = {}
        for bucket, (static, _, _) in runtime.cache.items():
            host = tuple(torch.empty(x.shape, dtype=x.dtype, pin_memory=True) for x in static)
            self.buffers[bucket] = (host, torch.cuda.Event(), False)

    def stage(self, request):
        bucket = next((b for b in self.buffers if b >= len(request['ids'])), None)
        if bucket is None:
            raise RuntimeError('Unprepared shape: candidate refuses online build/fallback')
        host, copied, pending = self.buffers[bucket]
        if pending and not copied.query():
            copied.synchronize()
        rgb, ids, picks, mask = host
        rgb.copy_(torch.from_numpy(request['rgb']))
        ids.zero_()
        ids[0, :len(request['ids'])].copy_(request['ids'])
        picks.copy_(request['picks'])
        mask.zero_()
        mask[0, :request['k']] = True
        static, graph, output = self.runtime.cache[bucket]
        for target, source in zip(static, host):
            target.copy_(source, non_blocking=True)
        copied.record()
        self.buffers[bucket] = (host, copied, True)
        return static, graph, output

    def run(self, request):
        _, graph, output = self.stage(request)
        graph.replay()
        return output, True

    def check_inputs(self, request):
        static, _, _ = self.stage(request)
        expected = self.runtime.args(request, static[1].shape[1])
        torch.cuda.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(static, expected)):
            raise RuntimeError('Transport changed model-facing inputs')
