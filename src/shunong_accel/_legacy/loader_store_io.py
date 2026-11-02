"""One lane's official Store I/O daemon; no inference in the daemon."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def load_state(config, profile):
    import sllm_store.torch as engine
    from sllm_store.client import SllmStoreClient
    port = 18073 + int(os.environ['SHUNONG_LANE'][1:])
    storage = Path(config['store_storage_path'])
    log_path = Path(os.environ['SHUNONG_OUTPUT'])/'slurm-store-service.out'
    t = time.perf_counter()
    command = [sys.executable, '-u', str(Path(__file__).with_name('loader_store_io.py')),
               '--serve', str(storage), str(port)]
    with log_path.open('w') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+90
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Store daemon exited {server.returncode}; see {log_path}')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Store daemon readiness timeout')
                    time.sleep(.1)
            profile['store_service_start_seconds'] = time.perf_counter()-t
            # The official low-level torch loader hardcodes port8073. Adapt
            # only the endpoint to permit independent lanes on one host.
            original = engine.SllmStoreClient
            engine.SllmStoreClient = lambda address: SllmStoreClient(f'127.0.0.1:{port}')
            try:
                t = time.perf_counter()
                state = engine.load_dict(config['store_model_path'], {'': 0}, str(storage))
                profile['store_read_and_h2d_seconds'] = time.perf_counter()-t
            finally:
                engine.SllmStoreClient = original
        finally:
            t = time.perf_counter()
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            profile['store_service_stop_seconds'] = time.perf_counter()-t
            profile['store_io_daemon_pid'] = server.pid
            profile['store_io_daemon_exitcode'] = server.returncode
    # All copies were confirmed complete before daemon shutdown. GPU storage
    # belongs to the client. No second PID remains during model inference/proof.
    return state


if __name__ == '__main__':
    assert sys.argv[1] == '--serve'
    from shunong_accel._legacy.gpu_guard import require_allocation
    require_allocation()
    import asyncio
    from sllm_store.server import serve
    asyncio.run(serve(host='127.0.0.1', port=int(sys.argv[3]), storage_path=sys.argv[2],
                      num_thread=4, chunk_size=32*2**20, mem_pool_size=12*2**30,
                      registration_required=False))
