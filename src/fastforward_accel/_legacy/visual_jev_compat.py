"""Only adapt newer image-type metadata calls to Transformers 4.57.

Upstream VDM sources stay unchanged. Single-image, no-video independent path
already derives image positions from input_ids in the older implementation.
"""
import inspect


def install_metadata_shim(vdm):
    model = vdm.vl_model
    original = model.get_rope_index
    if 'mm_token_type_ids' in inspect.signature(original).parameters:
        return {'applied': False}

    def rope(*args, mm_token_type_ids=None, **kwargs):
        return original(*args, **kwargs)

    def remove_unused_metadata(module, args, kwargs):
        return args, {k:v for k,v in kwargs.items() if k != 'mm_token_type_ids'}

    model.get_rope_index = rope
    model.register_forward_pre_hook(remove_unused_metadata, with_kwargs=True)
    return {'applied': True, 'reason': '4.57 derives image types from image-token IDs',
            'scope': 'single-image independent path; no architecture, weights, prompt or readout changes'}
