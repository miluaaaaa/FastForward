import argparse
import json
from .api import Accelerator


def main():
    p = argparse.ArgumentParser(description='FastForward visual choice inference')
    p.add_argument('--bundle', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--choices', nargs='+', required=True)
    p.add_argument('--backend', choices=['fast', 'native', 'established'], default='fast')
    a = p.parse_args()
    rt = Accelerator(a.bundle, backend=a.backend)
    print(json.dumps(rt.predict(a.image, a.question, a.choices), ensure_ascii=False))
