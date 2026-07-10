#!/usr/bin/env python
"""Scrub host-specific noise from an executed notebook, and nothing else.

The tutorial is rendered into public documentation, so absolute paths, site-package locations and
thread-pool chatter must not survive. An earlier version of this step dropped *every* stderr
stream, which also deleted the package's own truncation ``UserWarning`` -- the one disclosure the
tutorial exists to teach. Strip the offending lines; keep the warning.

Usage: ``python scrub_notebook.py tutorial.ipynb``
"""
from __future__ import annotations

import re
import sys

import nbformat

#: Lines that reveal the execution host, or are pure environment noise.
LEAK = re.compile(
    r"/fh/|/home/|/tmp/|site-packages|LOKY_|JAX_PLATFORMS|OMP_NUM_THREADS"
    r"|Kernel is running over TCP|IPKernelApp"
)


def scrub(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    dropped = kept = stripped = 0

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        keep_outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                lines = out.get("text", "").splitlines()
                clean = [line for line in lines if not LEAK.search(line)]
                stripped += len(lines) - len(clean)
                if not any(line.strip() for line in clean):
                    dropped += 1
                    continue
                out["text"] = "\n".join(clean) + "\n"
                kept += 1
            keep_outputs.append(out)
        cell["outputs"] = keep_outputs

    nbformat.write(nb, path)
    print(f"stderr streams: {dropped} dropped, {kept} kept; {stripped} leak lines stripped")

    # Fail loudly rather than publish a leak.
    remaining = [
        line
        for cell in nb.cells
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
        for line in out.get("text", "").splitlines()
        if LEAK.search(line)
    ]
    if remaining:
        print(f"ERROR: {len(remaining)} leak line(s) survived: {remaining[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(scrub(sys.argv[1] if len(sys.argv) > 1 else "tutorial.ipynb"))
