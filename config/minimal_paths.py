"""
Minimal path utilities for cross-platform environment detection and checkpoint directories.

- detect_platform(): 'kaggle' | 'colab' | 'jupyter' | 'script'
- get_root_dir(): repository root (assumes this file is under <root>/config)
- get_save_dir(platform=None): base directory to save outputs
- get_checkpoint_dir(platform=None): default checkpoints directory
- join_path(*parts): safe path join (str)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path


def detect_platform() -> str:
    """Detect execution environment.
    Returns: 'kaggle' | 'colab' | 'jupyter' | 'script'
    """
    # Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working'):
        return 'kaggle'
    # Colab
    try:
        import google.colab  # type: ignore
        return 'colab'
    except Exception:
        pass
    # Jupyter (rough check)
    try:
        from IPython import get_ipython  # type: ignore
        if get_ipython() is not None:
            return 'jupyter'
    except Exception:
        pass
    # Default script/IDE
    return 'script'


def get_root_dir() -> str:
    """Return repository root directory as string.
    Assumes this file lives in <root>/config.
    """
    here = Path(__file__).resolve()
    root = here.parent.parent
    return str(root)


def get_save_dir(platform: str | None = None) -> str:
    """Return base directory for saving outputs depending on platform."""
    platform = platform or detect_platform()
    if platform == 'kaggle':
        return '/kaggle/working'
    if platform == 'colab':
        return '/content'
    # local jupyter/script: use project-level output folder
    return str(Path(get_root_dir()) / 'output')


def get_checkpoint_dir(platform: str | None = None) -> str:
    """Return directory where checkpoints are stored.
    - Kaggle/Colab: prefer save dir root
    - Local: use <root>/checkpoints
    """
    platform = platform or detect_platform()
    if platform in ('kaggle', 'colab'):
        return get_save_dir(platform)
    return str(Path(get_root_dir()) / 'checkpoints')


def join_path(*parts: str) -> str:
    """Join path parts safely and return as string."""
    return str(Path(parts[0]).joinpath(*parts[1:]))


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    plat = detect_platform()
    root = get_root_dir()
    save = get_save_dir(plat)
    ckpt = get_checkpoint_dir(plat)
    print(f"Platform: {plat}")
    print(f"Root:     {root}")
    print(f"SaveDir:  {save}")
    print(f"CkptDir:  {ckpt}")
