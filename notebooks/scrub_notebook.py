#!/usr/bin/env python
"""Scrub host-specific noise from an executed notebook without destroying its warnings.

The tutorial renders into public documentation, so absolute paths and environment chatter must not
survive. Two earlier versions of this step got that wrong in opposite directions:

1. dropping **every** stderr stream also deleted the package's own truncation ``UserWarning`` --
   the one disclosure the tutorial exists to teach; and
2. dropping every *line* matching a host path deleted warnings too, because CPython prints
   ``<path>/steps.py:202: UserWarning: <message>`` **on a single line**. The message rides on the
   line that carries the path. Stripping the line decapitates the warning and leaves its orphaned
   source echo (``  warnings.warn(...)``) behind as a meaningless fragment.

So this scrubber classifies each stderr line:

* a warning raised **by this package** -> keep it, with the path redacted to ``spatial_smooth/…``;
* a warning raised by anything else -> drop it *and* its indented continuation lines;
* any other line carrying a host path or environment noise -> drop it;
* everything else -> keep.

Usage: ``python scrub_notebook.py tutorial.ipynb``
"""
from __future__ import annotations

import re
import sys

import nbformat

#: A CPython warning header: ``<path>:<lineno>: <Category>: <message>``.
WARNING_HEADER = re.compile(
    r"^(?P<path>.*?)(?P<module>[\w./\\-]*?[\w-]+\.py):(?P<lineno>\d+): "
    r"(?P<category>\w*Warning): (?P<message>.*)$"
)

#: Lines that reveal the execution host, or are pure environment noise.
LEAK = re.compile(
    r"/fh/|/home/|/tmp/|site-packages|LOKY_|JAX_PLATFORMS|OMP_NUM_THREADS"
    r"|Kernel is running over TCP|IPKernelApp"
)

#: Warnings worth publishing come from these modules. Everything else is third-party chatter
#: (tqdm's IProgress notice, docrep's SyntaxWarning, ...) that teaches the reader nothing.
OURS = ("steps.py", "smoothers.py", "core.py", "plot.py", "_deps.py")

#: A continuation of the preceding warning: CPython indents the offending source line.
CONTINUATION = re.compile(r"^\s+\S")


def scrub_text(text: str) -> str:
    """Return ``text`` with host paths removed and this package's warnings preserved."""
    out: list[str] = []
    dropping_continuation = False

    for line in text.splitlines():
        header = WARNING_HEADER.match(line)
        if header:
            module = header["module"].replace("\\", "/").rsplit("/", 1)[-1]
            if module in OURS:
                # Keep the warning; redact only the path that precedes it.
                out.append(
                    f"spatial_smooth/{module}:{header['lineno']}: "
                    f"{header['category']}: {header['message']}"
                )
                dropping_continuation = False
            else:
                dropping_continuation = True  # drop it and its source echo
            continue

        if dropping_continuation and CONTINUATION.match(line):
            continue
        dropping_continuation = False

        if LEAK.search(line):
            continue
        out.append(line)

    return "\n".join(out)


def scrub(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    kept = dropped = 0

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                cleaned = scrub_text(out.get("text", ""))
                if not cleaned.strip():
                    dropped += 1
                    continue
                out["text"] = cleaned + "\n"
                kept += 1
            outputs.append(out)
        cell["outputs"] = outputs

    nbformat.write(nb, path)
    print(f"stderr streams: {dropped} dropped, {kept} kept (paths redacted, warnings preserved)")

    # Fail loudly rather than publish a leak, or an orphaned warning fragment.
    survivors = [
        line
        for cell in nb.cells
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
        for line in out.get("text", "").splitlines()
        if LEAK.search(line)
    ]
    if survivors:
        print(f"ERROR: {len(survivors)} leak line(s) survived: {survivors[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(scrub(sys.argv[1] if len(sys.argv) > 1 else "tutorial.ipynb"))
