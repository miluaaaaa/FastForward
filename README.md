# FastForward

Engineering recipes and a model-specific inference runtime for faster vision-language inference.

FastForward combines compact visual inputs and prompts, FP8 computation, and CUDA graph execution. It is an engineering project with measured accuracy–latency trade-offs. The validated task is **single-image multiple-choice visual question answering**, not unrestricted text generation. Different model families require different adaptations.

## Results

Primary Qwen3-VL comparison cohort: four sizes × six benchmarks × eight methods = **192 cells**, **13,771 questions per model**, three timing repetitions, **1,322,016 calls**, NVIDIA H100.

| Qwen3-VL | Pooled speedup | Largest benchmark accuracy loss |
|---|---:|---:|
| 2B | 6.291× | 3.200 pp |
| 4B | 5.866× | 2.526 pp |
| 8B | 5.378× | 2.859 pp |
| 32B | 4.697× | 3.133 pp |

- [Complete English tables (PDF)](results/FASTFORWARD_RESULTS.pdf)
- [All 192 comparison cells (CSV)](results/qwen-main.csv)
- [Cross-family results (CSV)](results/cross-family.csv)
- [Measurement details and limitations](RESULTS.md)

These are archived measurements, not fresh tests of this sanitized release. Speedup compares native BF16/native inputs with accelerated input/prompt approximations and FP8, not identical-input kernel-only optimization. Timing is steady-state read-image-to-JSON latency, excluding model loading, weight packing and initial graph capture. Public API validation/probability output adds overhead not included in the measured internal interface.

## Runtime status

The source is derived from the existing **0.2.4 family plugin**. Its separately recorded standalone speedups were 6.926× / 6.067× / 5.584× / 4.625×; these are a different cohort and must not replace the table above. This release preserves the core runtime arithmetic and makes packaging/path changes. No new GPU parity or performance claim is made for the release.

Supported public path: Qwen3-VL 2B, 4B, 8B, 32B, single image, 2–8 answer choices, one visible exclusively allocated Hopper GPU; tested hardware was H100. No CPU fallback. Dependencies and numerical behavior are version-sensitive; see [tested environment](tested-environment.txt).

**This is source plus results, not a self-contained turnkey binary.** Base-model weights, zero-repair checkpoints and calibration profiles are external. The four config templates retain expected artifact hashes; users must provide the matching artifacts and set local paths. Merely changing a model name or architecture is unsupported. Other-family results are included as supporting studies; their adapters are not part of this Qwen-focused release.

## Installation and invocation

Install matching CUDA PyTorch wheels first. Then:

```bash
python -m pip install .
```

Edit a template in `configs/` to set `model`, `checkpoint` and `profile_path` to existing local assets. Preserve the matching SHA256 fields. The loader uses local files rather than downloading models. Profiles and checkpoints are **not** bundled or currently hosted by this repository; running fast mode requires obtaining/recreating compatible artifacts. `configs/` is a record of configuration requirements, not a claim that those assets are available for download.

Inside a suitable GPU allocation:

```python
from shunong_accel import Accelerator

engine = Accelerator("configs/4B.json", backend="fast")
result = engine.predict("image.png", "Which object is larger?", ["The cup", "The plate"])
print(result["answer"])
```

Or use `shunong-accel --help` for the command-line interface. Fast mode releases BF16 projection weights: create a separate native instance for a later native comparison. Initial capture occurs on first use. The runtime serializes requests and does not implement a general multi-user server.

The namespaced `_legacy` modules are retained implementation dependencies, not a promise that every historical experimental script is a supported interface. Some historical loader/evidence helpers assume Slurm and H100. Scheduler-specific automation is not distributed as a required launcher.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

These tests check CPU-side request validation, response behavior and archived table coverage. They do not run a model or establish GPU performance. For any new benchmark, verify real same-process GPU execution and scoring semantics before collecting results.

## License

Project source: Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party libraries, models and datasets retain their own licenses. No weights, benchmark media, per-example predictions, secrets or private cluster configuration are included. Comparator source implementations are not redistributed.
