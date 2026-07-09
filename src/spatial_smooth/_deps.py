"""Lazy, actionable dependency handling.

Heavy or optional third-party packages are never imported at module import time.
:func:`require` imports on demand and, when the package is absent, raises an
``ImportError`` that names the package, says what it is needed for, and gives the
exact ``pip install`` line.
"""
from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple

__all__ = ["require", "have", "check_dependencies", "DEPENDENCIES", "OPTIONAL"]

#: ``package -> (import name, pip target, what it is used for)``
DEPENDENCIES: Dict[str, Tuple[str, str, str]] = {
    "numpy": ("numpy", "numpy>=1.23", "core arrays (required)"),
    "pandas": ("pandas", "pandas>=1.5", "core tables (required)"),
    "scipy": ("scipy", "scipy>=1.10", "kd-tree neighbours + grid interpolation (required)"),
    "anndata": ("anndata", "anndata>=0.9", "the AnnData container (required)"),
    "kompot": ("kompot", "kompot>=0.8.0", "GP smoothing over a cell-state embedding (step `KompotGP`)"),
    "palantir": ("palantir", "palantir>=1.4", "diffusion-map embedding (`compute_diffusion_map`)"),
    "scanpy": ("scanpy", "scanpy>=1.9", "plotting backend `scanpy` (`spatial_smooth.pl`)"),
    "squidpy": ("squidpy", "squidpy>=1.4", "plotting backend `squidpy` (`spatial_smooth.pl`)"),
    "KDEpy": ("KDEpy", "KDEpy>=1.1", "fine-grid FFT-KDE smoother (step `Kde`)"),
    "matplotlib": ("matplotlib", "matplotlib>=3.6", "figure canvas for the plotting backends"),
}

#: Dependencies whose absence is not an error until the feature that needs them is used.
OPTIONAL = frozenset({"kompot", "palantir", "scanpy", "squidpy", "KDEpy", "matplotlib"})


def require(pkg: str):
    """Import ``pkg``, or raise an ``ImportError`` that says exactly how to fix it.

    Parameters
    ----------
    pkg
        Key into :data:`DEPENDENCIES`.

    Raises
    ------
    ImportError
        With the package name, its purpose, and the ``pip install`` line.
    """
    try:
        import_name, pip_target, purpose = DEPENDENCIES[pkg]
    except KeyError:  # pragma: no cover - programming error
        raise KeyError(f"{pkg!r} is not a known spatial_smooth dependency") from None
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        raise ImportError(
            f"spatial_smooth needs `{pkg}` for: {purpose}.\n"
            f'    Install it with:  pip install "{pip_target}"\n'
            f"(original error: {exc})"
        ) from exc


def _version_of(module, import_name: str) -> str:
    """Distribution version, preferring the metadata over a (sometimes deprecated) attribute."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(import_name)
    except PackageNotFoundError:
        return str(getattr(module, "__version__", "?"))


def have(pkg: str) -> bool:
    """``True`` if ``pkg`` can be imported right now (never raises)."""
    import_name = DEPENDENCIES[pkg][0]
    try:
        importlib.import_module(import_name)
    except ImportError:
        return False
    return True


def check_dependencies(verbose: bool = True) -> Dict[str, Optional[str]]:
    """Report which dependencies are importable and at which version.

    Prints a table and returns ``{package: version or None}``. Call it at the top of a
    script or notebook to surface gaps *before* the analysis starts, each with the exact
    ``pip install`` line that fixes it.
    """
    from . import __version__

    status: Dict[str, Optional[str]] = {}
    rows = []
    for pkg, (import_name, pip_target, purpose) in DEPENDENCIES.items():
        try:
            mod = importlib.import_module(import_name)
            ver = _version_of(mod, import_name)
            status[pkg] = ver
            rows.append((pkg, ver, "ok", purpose))
        except ImportError:
            status[pkg] = None
            state = f'missing: pip install "{pip_target}"'
            rows.append((pkg, "-", state, purpose))

    if verbose:
        width = max(len(r[0]) for r in rows)
        print(f"spatial_smooth v{__version__} -- dependency check")
        for pkg, ver, state, _purpose in rows:
            tag = "" if pkg not in OPTIONAL else " (optional)"
            print(f"  {pkg:<{width}}  {ver:<10}  {state}{tag}")
        missing_required = [p for p, v in status.items() if v is None and p not in OPTIONAL]
        missing_optional = [p for p, v in status.items() if v is None and p in OPTIONAL]
        print()
        if missing_required:
            print(f"  Required packages missing: {', '.join(missing_required)}.")
        elif missing_optional:
            print(f"  All required packages present (optional missing: {', '.join(missing_optional)}).")
        else:
            print("  All dependencies present.")
    return status
