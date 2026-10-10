from __future__ import annotations

import os
import re
from pathlib import Path
import yaml

# Matches ${VAR} or ${VAR:-default} in the raw YAML text, expanded against the
# process environment before parsing - lets config values like dataset roots be
# overridden per-machine (e.g. DFB_DATA_ROOT=/data/dfb python3 main_dfb.py ...)
# without editing the checked-in config, while still working unmodified when the
# env var isn't set, via the :-default fallback.
_ENV_VAR_RE = re.compile(r'\$\{(\w+)(:-(.*?))?\}')


def _expand_env_vars(text: str) -> str:
    def repl(match: re.Match) -> str:
        name, has_default, default = match.group(1), match.group(2), match.group(3)
        if name in os.environ:
            return os.environ[name]
        return default if has_default is not None else match.group(0)
    return _ENV_VAR_RE.sub(repl, text)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as fh:
        raw = fh.read()
    return yaml.safe_load(_expand_env_vars(raw))


def ensure_records_layout(project_root: Path, cfg: dict) -> Path:
    records_dir = project_root / cfg.get('project', {}).get('records_dir', 'records')
    records_dir.mkdir(parents=True, exist_ok=True)
    # Everything else (models_<variant>/, eval/<dataset>_<variant>/, results/, index/,
    # ablation/) is created on demand by the code that writes into it - no need to
    # pre-create a fixed directory list here.
    for rel in ['logs', 'results']:
        (records_dir / rel).mkdir(parents=True, exist_ok=True)
    return records_dir
