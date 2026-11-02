"""Public API; model artifacts and hardware capabilities are explicit."""
import hashlib
import json
import math
from pathlib import Path
import threading


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_request(image, question, candidates):
    if not Path(image).is_file():
        raise FileNotFoundError(image)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('A nonempty question is required')
    if not isinstance(candidates, (list, tuple)) or not 2 <= len(candidates) <= 8:
        raise ValueError('Provide two to eight choices')
    if any(not isinstance(x, str) or not x.strip() for x in candidates):
        raise ValueError('Choices must be nonempty strings')


def response(logits, candidates, item_id):
    if len(logits) != len(candidates) or not all(math.isfinite(x) for x in logits):
        raise RuntimeError('Invalid model output')
    prediction = max(range(len(logits)), key=lambda i: logits[i])
    exps = [math.exp(x - max(logits)) for x in logits]
    return dict(id=item_id, answer=dict(letter=chr(65+prediction), text=candidates[prediction]),
                logits=logits, probabilities=[x/sum(exps) for x in exps])


class Accelerator:
    """Load one supported model and its verified adaptation bundle.

    The caller owns a CUDA device. Cold graph capture is included on first use.
    Instances serialize calls; graph buffers are never shared across requests.
    """
    def __init__(self, bundle, *, backend='fast'):
        self.bundle = Path(bundle).resolve()
        c = json.loads(self.bundle.read_text())
        if c.get('schema_version') != 1 or c.get('task') != 'visual_choice':
            raise ValueError('Unsupported adaptation bundle schema or task')
        if backend not in ('fast', 'native', 'established'):
            raise ValueError('Unknown backend')
        for key in ('model', 'checkpoint'):
            if not Path(c[key]).exists():
                raise FileNotFoundError(c[key])
        if sha256(c['checkpoint']) != c['checkpoint_sha256']:
            raise ValueError('Recovery checkpoint SHA256 mismatch')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU is required; CPU fallback is disabled')
        if torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError('This FP8 backend requires Hopper or newer hardware')
        torch.set_num_threads(1)
        self.backend = backend
        self.lock = threading.RLock()
        if c['architecture'] == 'qwen3_vl_family':
            from .family_qwen import FamilyQwen
            self.runtime = FamilyQwen(c, backend)
        else:
            raise ValueError('No adapter registered for this architecture')
        self.config = c
        self.model = self.runtime.model

    def predict(self, image, question, candidates, item_id='request', *, backend=None):
        validate_request(image, question, candidates)
        mode = self.backend if backend is None else backend
        if mode not in ('fast', 'native', 'established'):
            raise ValueError('Unknown backend')
        import torch
        context = torch.no_grad if self.config['architecture'] == 'qwen3_vl_family' else torch.inference_mode
        with self.lock, context():
            z = self.runtime.predict(str(image), question, list(candidates), item_id, mode)
        return response(z, candidates, item_id)

    def warmup(self, requests):
        for r in requests:
            self.predict(r['image_path'], r['instruction'], r['candidates'], r.get('item_id', 'warmup'))

    def activate_fast(self):
        """Convert a scaled Qwen native instance to FP8 outside timed requests.

        With retain_bf16=False this releases BF16 linear weights permanently.
        Subsequent native calls need a separately loaded native instance.
        """
        with self.lock:
            if self.config['architecture'] not in ('qwen3_vl_scaled', 'qwen3_vl_family'):
                raise RuntimeError('Residency transitions are supported only by scaled Qwen bundles')
            import torch
            with torch.no_grad():
                self.runtime.activate_fast()
                torch.cuda.synchronize()
            self.backend = 'fast'

