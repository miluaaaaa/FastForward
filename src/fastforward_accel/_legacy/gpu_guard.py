"""Fail-closed same-PID, same-physical-GPU forward-pass evidence."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch


def command(args):
    return subprocess.check_output(args, text=True, timeout=8, stderr=subprocess.DEVNULL)


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU model work must execute inside a Slurm allocation routed by Zen")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 1 or not devices[0].strip() or torch.cuda.device_count() != 1:
        raise RuntimeError("One lane must have exactly one allocated CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name:
        raise RuntimeError("This comparison requires the requested H100")
    return name


def prove(model, forward, output: Path):
    require_allocation()
    pid = os.getpid()
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    indices = {}
    for line in command(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]).splitlines():
        index, value = [x.strip() for x in line.split(",")]
        indices[index] = value
    minimum = max(2048, int(sum(p.numel() * p.element_size() for p in model.parameters() if p.is_cuda) / 2**20 * .65))
    samples, stop = [], threading.Event()

    def collect():
        while not stop.is_set():
            try:
                start = time.time()
                apps = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"])
                pmon = command(["nvidia-smi", "pmon", "-s", "um", "-c", "1"])
                samples.append(dict(start=start, end=time.time(), apps=apps, pmon=pmon))
            except Exception as error:
                samples.append(dict(error=type(error).__name__))
            stop.wait(.1)

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    start = time.time()
    iterations = 0
    try:
        # These are actual model warmup forwards, never a synthetic CUDA load.
        while time.time() - start < 8 or iterations == 0:
            forward()
            torch.cuda.synchronize()
            iterations += 1
    finally:
        stop.set()
        thread.join(timeout=10)
    max_memory, max_compute = 0, 0
    collisions = set()
    for sample in samples:
        for line in sample.get("apps", "").splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) == 3 and fields[0] == uuid:
                if fields[1] == str(pid):
                    max_memory = max(max_memory, int(fields[2]))
                else:
                    collisions.add(fields[1])
        for line in sample.get("pmon", "").splitlines():
            fields = line.split()
            if len(fields) >= 5 and indices.get(fields[0]) == uuid and fields[1] == str(pid):
                try:
                    max_compute = max(max_compute, int(fields[3]))
                except ValueError:
                    pass
    evidence = dict(pid=pid, gpu_uuid=uuid, minimum_model_memory_mib=minimum,
                    memory_mib=max_memory, sm_percent=max_compute, other_pids=sorted(collisions),
                    warmup_forwards=iterations, samples=samples,
                    passed=max_memory >= minimum and max_compute > 0 and not collisions)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gpu_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not evidence["passed"]:
        raise RuntimeError("Missing exclusive same-PID GPU memory/compute evidence during real forward")
    return evidence


def invalidate(output: Path, error):
    output.mkdir(parents=True, exist_ok=True)
    (output / "INVALID.json").write_text(json.dumps(dict(status="INVALID", error=str(error), pid=os.getpid())) + "\n")
    if os.environ.get("SLURM_JOB_ID"):
        subprocess.run(["scancel", os.environ["SLURM_JOB_ID"]], check=False)
