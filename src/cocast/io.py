"""Small file helpers shared by command-line entry points."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_real(path):
    """Select baseline columns from SPSS, or require the declared CSV schema."""
    import pandas as pd
    from . import schema
    path = Path(path)
    if path.suffix.lower() == '.sav':
        import pyreadstat
        frame, _ = pyreadstat.read_sav(str(path), usecols=schema.REAL_COLUMNS)
    else:
        frame = pd.read_csv(path)
    schema.validate_real_table(frame)
    return frame[schema.REAL_COLUMNS].reset_index(drop=True)
