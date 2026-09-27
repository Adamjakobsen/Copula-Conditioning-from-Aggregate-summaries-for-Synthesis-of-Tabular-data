"""Manage a CUDA vLLM server without touching unrelated processes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from .generation import check_context
from .prompts import load_bundle


def check_server(settings):
    """Verify the advertised model on an explicitly supplied server."""
    with urlopen(settings['base_url'].rstrip('/') + '/models', timeout=10) as response:
        models = json.load(response)
    if settings['model'] not in {m['id'] for m in models.get('data', [])}:
        raise ValueError('The server does not advertise the configured model.')


@contextmanager
def serve(settings, bundles, log_path, *, tensor_parallel_size=1, startup_timeout=1200):
    """Start and stop only the process group created by this context."""
    from huggingface_hub import snapshot_download
    import torch
    from urllib.parse import urlparse

    url = urlparse(settings['base_url'])
    if url.hostname != '127.0.0.1' or not url.port:
        raise ValueError('Managed serving requires a 127.0.0.1 URL with an explicit port.')
    if tensor_parallel_size < 1 or torch.cuda.device_count() < tensor_parallel_size:
        raise ValueError('Insufficient visible CUDA GPUs for the requested tensor parallel size.')
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1', url.port)) == 0:
            raise RuntimeError('Port occupied. Choose another port or explicitly use --existing-server.')
    checkpoint = snapshot_download(settings['model'], revision=settings['model_revision'])
    for bundle in bundles:
        _, records = load_bundle(bundle)
        check_context(records, {**settings, 'tokenizer_path': checkpoint})
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', checkpoint, '--served-model-name', settings['model'],
               '--host', '127.0.0.1', '--port', str(url.port), '--dtype', 'bfloat16',
               '--tensor-parallel-size', str(tensor_parallel_size),
               '--max-model-len', str(settings['context_window']), '--max-num-seqs', '4',
               '--max-num-batched-tokens', '2048', '--gpu-memory-utilization', '0.90',
               '--no-enable-prefix-caching', '--language-model-only', '--generation-config', 'vllm',
               '--default-chat-template-kwargs', '{"enable_thinking":false}']
    environment = {**os.environ, 'VLLM_USE_FLASHINFER_SAMPLER': '0', 'VLLM_NO_USAGE_STATS': '1',
                   'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            print(f'Starting {settings["model"]}. Server log: {log_path}', flush=True)
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'vLLM exited during startup. Inspect {log_path}.')
                try:
                    check_server(settings)
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError(f'vLLM startup timed out. Inspect {log_path}.')
            yield
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
