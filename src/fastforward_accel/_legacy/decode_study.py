"""CPU-only, fresh-file image-path ablation; never reports model latency.

Output equality is checked against the selected sub10-v6 image path. The
reference pixels are used only for verification, never as candidate inputs.
"""
import argparse
import ast
import collections
import hashlib
import io
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def original(path, side, benchmark):
    if benchmark == 'AI2D':
        x = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
        assert x is not None
        return cv2.cvtColor(cv2.resize(x, (side, side), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    with Image.open(path) as image:
        image.draft('RGB', (side, side))
        return np.array(image.resize((side, side), Image.Resampling.NEAREST).convert('RGB'), copy=True)


def candidate(path, side, method, scratch):
    """Every method opens its own original file inside its timed call."""
    t0 = time.perf_counter_ns()
    if method in ('control_ai', 'control_mm'):
        out = original(path, side, 'AI2D' if method == 'control_ai' else 'MMStar')
        return out, dict(total_ms=(time.perf_counter_ns()-t0)/1e6)
    with open(path, 'rb') as f:
        data = f.read()
    t1 = time.perf_counter_ns()
    if method.startswith('cv_'):
        x = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        assert x is not None
        t2 = time.perf_counter_ns()
        if method == 'cv_reuse':
            small, out = scratch
            cv2.resize(x, (side, side), dst=small, interpolation=cv2.INTER_LINEAR)
            t3 = time.perf_counter_ns()
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=out)
        else:
            small = cv2.resize(x, (side, side), interpolation=cv2.INTER_LINEAR)
            t3 = time.perf_counter_ns()
            out = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    elif method == 'pil_split':
        with Image.open(io.BytesIO(data)) as image:
            image.draft('RGB', (side, side))
            image.load()
            t2 = time.perf_counter_ns()
            small = image.resize((side, side), Image.Resampling.NEAREST)
            t3 = time.perf_counter_ns()
            out = np.array(small.convert('RGB'), copy=True)
    else:
        if method.startswith('spng'):
            import pyspng
            x = pyspng.load(data)
        elif method.startswith('imagecodecs'):
            import imagecodecs
            x = imagecodecs.png_decode(data)
        elif method == 'pil_rgb_reduce':
            with Image.open(io.BytesIO(data)) as image:
                x = np.array(image.convert('RGB'), copy=True)
        else:
            raise ValueError(method)
        assert x.dtype == np.uint8 and x.ndim == 3 and x.shape[2] == 3, ('unsupported_pixel_mode', x.shape, str(x.dtype))
        # OpenCV's reduced-PNG path decodes full resolution and then uses
        # INTER_LINEAR_EXACT to floor(width/4), floor(height/4). The equality
        # gate below verifies this behavior; it is not assumed equivalent.
        t2 = time.perf_counter_ns()
        h, w = x.shape[:2]
        reduced = cv2.resize(x, (w//4, h//4), interpolation=cv2.INTER_LINEAR_EXACT)
        out = cv2.resize(reduced, (side, side), interpolation=cv2.INTER_LINEAR)
        t3 = time.perf_counter_ns()
    t4 = time.perf_counter_ns()
    assert out.dtype == np.uint8 and out.shape == (side, side, 3) and out.flags.c_contiguous
    return out, dict(read_ms=(t1-t0)/1e6, decode_ms=(t2-t1)/1e6,
                     resize_ms=(t3-t2)/1e6, color_copy_ms=(t4-t3)/1e6,
                     total_ms=(t4-t0)/1e6)


def main(args):
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    info = dict(workstream='FastForward', task='image-path-only',
                node=platform.node(), cpu=platform.processor(), pid=os.getpid(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cv2=cv2.__version__, numpy=np.__version__, pillow=Image.__version__,
                no_gpu_requested=True, no_model_loaded=True, os_cache_cleared=False,
                timing='Fresh original file open through contiguous RGB array. Imports, validation and hashes excluded. No model or feature inference.')
    (root/'environment.json').write_text(json.dumps(info, indent=2)+'\n')
    available = ['cv_split', 'cv_reuse', 'pil_rgb_reduce']
    unavailable = {}
    for module, method in [('pyspng', 'spng_rgb_reduce'), ('imagecodecs', 'imagecodecs_rgb_reduce')]:
        try:
            m = __import__(module)
            available.append(method)
            info[module] = getattr(m, '__version__', 'unknown')
        except Exception as error:
            unavailable[module] = repr(error)
    (root/'environment.json').write_text(json.dumps(info, indent=2)+'\n')
    results = {}
    # Bind the local comparison implementation to the actual frozen runtime,
    # without importing torch or allocating a GPU for an image-only study.
    frozen_path = args.previous/'source/sub10_runtime.py'
    frozen_tree = ast.parse(frozen_path.read_text())
    decode_fn = next(n for n in frozen_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decode')
    frozen_scope = dict(cv2=cv2, np=np, Image=Image)
    exec(compile(ast.Module(body=[decode_fn], type_ignores=[]), str(frozen_path), 'exec'), frozen_scope)
    info['frozen_runtime_sha256'] = digest(frozen_path.read_bytes())
    rng = random.Random(118)
    with (root/'timings.jsonl').open('w') as output:
        for lane, bench, side in [('L1', 'MMStar', 192), ('L3', 'AI2D', 64)]:
            config_path = args.previous/lane/'config.json'
            c = json.loads(config_path.read_text())
            raw = Path(c['inference']).read_bytes()
            assert digest(raw) == c['dataset_hashes'][c['inference']]
            rows = [json.loads(s) for s in raw.splitlines()]
            indices = c['selected_indices']
            methods = ['control_ai']+available if bench == 'AI2D' else ['control_mm', 'pil_split']
            refs, assets, gates = {}, [], {m:dict(passed=True, checked=0, mismatches=[]) for m in methods}
            scratch = (np.empty((side, side, 3), np.uint8), np.empty((side, side, 3), np.uint8))
            for ix in indices:
                r = rows[ix]
                p = Path(r['image_path']) if bench == 'MMStar' else Path(c['data_root'])/r['image_relpath']
                data = p.read_bytes()
                assert digest(data) == r['image_sha256']
                reference = original(p, side, bench)
                assert np.array_equal(reference, frozen_scope['decode'](p, side, 'cvreduced' if bench == 'AI2D' else 'pilnearest'))
                with Image.open(io.BytesIO(data)) as image:
                    asset = dict(index=ix, item_id=r.get('item_id', r.get('source_record_id')), path=str(p),
                                 image_sha256=r['image_sha256'], bytes=len(data), format=image.format,
                                 width=image.width, height=image.height, mode=image.mode)
                assets.append(asset)
                refs[ix] = reference
                for method in methods:
                    try:
                        actual, _ = candidate(p, side, method, scratch)
                        exact = np.array_equal(actual, reference)
                        gate = gates[method]
                        gate['checked'] += 1
                        if not exact:
                            gate['passed'] = False
                            gate['mismatches'].append(dict(index=ix, max_abs=int(np.abs(actual.astype(np.int16)-reference).max()),
                                                           differing_channels=int(np.count_nonzero(actual != reference))))
                    except Exception as error:
                        gates[method]['passed'] = False
                        gates[method]['mismatches'].append(dict(index=ix, error=repr(error)))
            # Timing methods interleaved per row, and shuffled per repetition to
            # avoid giving every candidate the same favorable execution order.
            records = []
            valid = [m for m in methods if gates[m]['passed']]
            for repeat in range(5):
                for asset in assets:
                    order = list(valid)
                    rng.shuffle(order)
                    for method in order:
                        actual, timing = candidate(Path(asset['path']), side, method, scratch)
                        assert np.array_equal(actual, refs[asset['index']]), (method, asset['index'])
                        row = dict(benchmark=bench, method=method, repeat=repeat,
                                   index=asset['index'], item_id=asset['item_id'], **timing)
                        output.write(json.dumps(row)+'\n')
                        records.append(row)
            summary = {}
            for method in valid:
                recs = [r for r in records if r['method'] == method]
                summary[method] = dict(count=len(recs), unique=len({r['item_id'] for r in recs}),
                                       stages={key:summarize([r[key] for r in recs]) for key in recs[0] if key.endswith('_ms')},
                                       repeat_means_ms=[statistics.mean(r['total_ms'] for r in recs if r['repeat']==j) for j in range(5)])
            results[bench] = dict(side=side, unique=len(indices), methods=summary, gates=gates,
                                  formats=dict(collections.Counter(a['format'] for a in assets)), assets=assets,
                                  source_config_sha256=digest(config_path.read_bytes()))
            print(bench, json.dumps({m:x['stages']['total_ms'] for m,x in summary.items()}), flush=True)
    (root/'summary.json').write_text(json.dumps(dict(environment=info, unavailable=unavailable, benchmarks=results), indent=2)+'\n')
    print('COMPLETE CPU image path only', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=Path('/path/to/previous-run'))
    main(p.parse_args())
