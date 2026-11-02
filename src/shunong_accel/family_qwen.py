"""Public adapter for the frozen full-depth family configurations."""
import torch
from ._family.family_api import Accelerator as Measured

class FamilyQwen:
    def __init__(self, config, backend):
        if backend not in ('native', 'fast'):
            raise ValueError('Family bundles support native and fast backends')
        if config['execution_mode'] not in ('fp8', 'precise'):
            raise ValueError('Unverified execution mode')
        with torch.no_grad():
            self.engine = Measured(config, fast=backend == 'fast')
            self.engine.rt.select(config['variant'])
        self.model = self.engine.model

    def activate_fast(self):
        if not self.engine.fast:
            with torch.no_grad():self.engine.activate_fast()

    def predict(self, image, question, candidates, item_id, mode):
        result, _ = self.engine.predict(image, question, candidates, item_id, mode)
        return result['logits']
