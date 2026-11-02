# Experimental results and scope

The primary cohort is the complete 192-cell Qwen3-VL comparison. `results/qwen-main.csv` contains model, benchmark, method, accuracy_percent, loss_pp, mean_ms and speedup. Accuracy loss is native minus method accuracy; a negative value is an increase. Repetitions are timing replicates, not independent training seeds. Native is freshly loaded after accelerated methods execute sequentially on the same GPU/process.

Benchmarks: MMStar 1,500; AI2D 3,088; MMBench-EN 4,377; CV-Bench 2,638; VisualPuzzles 1,168; VisuLogic 1,000. MMBench uses provided-pass group accuracy, **not complete official CircularEval** (53 groups incomplete). CV-Bench uses source-balanced accuracy. Other metrics are forced-choice exact accuracy.

Comparators F3A, VisionZip, DivPrune, ReplaceMe, FastV and S2Prune are local Qwen adaptations sharing an accelerated backend, not their authors' original end-to-end systems. Budgets were chosen using 32 image-disjoint unlabeled GQA calibration examples. Several small-model settings retain only 2% of visual tokens. F3A and some S2Prune settings did not match the target ±10% latency interval. Large accuracy drops do not establish that the original methods are intrinsically ineffective. Full budgets and caveats appear in the PDF.

## Other-family studies

| Family | Scale | Speedup / throughput ratio | Scope |
|---|---|---|---|
| Ministral3 | 3B / 8B / 14B | 3.2318 / 3.5799 / 3.7294× | Request-stream timing includes capture |
| Gemma3 | 4B / 12B / 27B | 3.255639 / 2.555902 / 1.987976× | Batch-two throughput versus serial HF |
| InternVL3.5 | 8B | 1.64–1.82× | Per-benchmark warm latency; historical fixed-image controls |
| LLaVA-1.5 | 7B | 3.38× | GQA64 pilot only |

The CSV contains 42 full-benchmark comparisons, excluding the LLaVA pilot. These studies change several factors jointly and are transfer/adaptation evidence, not controlled component ablations. Their different timing boundaries and batch sizes prohibit averaging or ranking speedups across families. Missing latencies are not inferred from rounded speedups. Earlier invalid runs are not counted.

## Remaining limitations

Component attribution requires matched-input/matched-precision ablations. General generation claims need multi-token measurements. Profiles and external checkpoints remain a reproducibility dependency. This source release has only packaging/static and CPU-contract checks; it has not been rerun on GPU after sanitization. The preserved plugin and primary comparator cohort have different evaluation records.
