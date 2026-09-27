import importlib.metadata
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import REVISION, schema
from .io import atomic_text, read_real, sha256, write_json


def run_baseline(real_path, output_dir, *, method, n=None, epochs=500, seed=42, device='cpu'):
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    import torch

    if method not in {'ctgan', 'tvae'} or epochs < 1 or (n is not None and n < 1):
        raise ValueError('Use ctgan or tvae and positive epochs/row counts.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable. Use --device cpu for CPU execution.')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Baseline directory is not empty. Use a new run directory.')
    real = read_real(real_path)
    training = real.copy()
    categorical = [schema.SEX_COL, *schema.ALL_ITEM_COLUMNS]
    for col in categorical:
        training[col] = training[col].astype('Int64').astype('string')
    metadata = Metadata.detect_from_dataframe(data=training, table_name='table', infer_keys=None)
    metadata.update_columns(column_names=categorical, sdtype='categorical', table_name='table')
    metadata.update_column(column_name=schema.AGE_COL, sdtype='numerical', table_name='table')
    metadata.validate_table(data=training, table_name='table')
    # Seed before constructing and fitting either model. Do not offset TVAE's seed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    cls = CTGANSynthesizer if method == 'ctgan' else TVAESynthesizer
    model = cls(metadata, epochs=epochs, enable_gpu=device == 'cuda', verbose=False,
                enforce_min_max_values=True, enforce_rounding=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'revision': REVISION, 'kind': 'baseline', 'method': method, 'status': 'running',
                'seed': seed, 'regime': 'natural', 'epochs': epochs, 'device': device,
                'n_real': len(real), 'n_patients': n or len(real), 'real_sha256': sha256(real_path),
                'sdv_version': importlib.metadata.version('sdv'),
                'torch_version': torch.__version__, 'metadata': metadata.to_dict(),
                'age_missing_count': int(real[schema.AGE_COL].isna().sum())}
    write_json(out / 'manifest.json', manifest)
    print(f'Training {method.upper()}: {len(real)} participants, {epochs} epochs, {device}')
    try:
        model.fit(training)
        sampled = model.sample(num_rows=n or len(real), output_file_path=None)
        for col in categorical:
            sampled[col] = pd.to_numeric(sampled[col], errors='raise')
        sampled[schema.AGE_COL] = pd.to_numeric(sampled[schema.AGE_COL], errors='raise').round()
        sampled = sampled[schema.REAL_COLUMNS]
        schema.validate_real_table(sampled)
        atomic_text(out / 'dataset.csv', sampled.to_csv(index=False))
        manifest.update(status='complete', dataset_sha256=sha256(out / 'dataset.csv'))
        write_json(out / 'manifest.json', manifest)
    except BaseException:
        manifest['status'] = 'interrupted_or_failed'
        write_json(out / 'manifest.json', manifest)
        raise
    print(f'Completed {method.upper()} -> {out}')
