"""Default response-workflow loader: the validated Tensorizer GPU path."""
import json
import sys
from pathlib import Path


def load_model(config):
    from shunong_accel._legacy.gpu_guard import require_allocation
    require_allocation()
    c = dict(config)
    if c.get('loader_backend', 'original') == 'original':
        from .loader_study_runtime import load_profile
        return load_profile(c, 'original')[:2]
    checkpoint = Path(c.get('tensorizer_file', Path(c['model']) /
                          'loader_formats/tensorizer-2.12.1/model.tensors'))
    conversion = checkpoint.parent/'conversion.json'
    if not checkpoint.is_file() or not conversion.is_file():
        raise FileNotFoundError(f'Validated Tensorizer checkpoint required: {checkpoint}')
    report = json.loads(conversion.read_text())
    if not (report['passed'] and report['format'] == 'tensorizer-2.12.1'
            and Path(report['source_model']).resolve() == Path(c['model']).resolve()
            and any(f['path'] == checkpoint.name and f['bytes'] == checkpoint.stat().st_size
                    for f in report['files'])):
        raise ValueError('Tensorizer conversion metadata does not match the requested base model')
    c['tensorizer_file'] = str(checkpoint)
    deps = Path(c.get('tensorizer_deps',
        '/path/to/tensorizer-deps'))
    if deps.is_dir() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    from shunong_accel._legacy.loader_study_runtime import load_profile
    model, head, profile = load_profile(c, 'tensorizer')
    model.loading_backend = 'tensorizer'
    model.loading_profile = profile
    return model, head
