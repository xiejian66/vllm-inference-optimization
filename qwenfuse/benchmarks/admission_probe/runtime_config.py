"""Paths for a relocatable experiment workspace; no GPU imports."""
import os
from pathlib import Path
ROOT = Path(os.environ.get('QWENFUSE_WORKSPACE', '.qwenfuse-work')).expanduser().resolve()
os.environ.setdefault('QWENFUSE_WORKSPACE', str(ROOT))
MODEL = Path(os.environ.get('QWENFUSE_MODEL', str(ROOT / 'models/Qwen3-8B'))).expanduser().resolve()
os.environ.setdefault('QWENFUSE_MODEL', str(MODEL))
REPOS = {k: Path(os.environ.get('QWENFUSE_REPO_' + v, str(ROOT / 'src' / v))).expanduser().resolve()
         for k,v in {'A4':'A','B4':'B','C4':'C','S0':'A','S1':'D'}.items()}
