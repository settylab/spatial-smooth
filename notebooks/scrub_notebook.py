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

3. whitelisting warnings by the **file** in their header deletes ours whenever a user raises them
   from a notebook cell, because ``stacklevel=2`` attributes the header to the *caller*.

So this scrubber classifies each stderr line by the warning's **category**, which no stacklevel can
change:

* a warning whose category is one of ours (:class:`spatial_smooth.SpatialSmoothWarning` and its
  subclasses) -> keep it, with the leading path redacted;
* a warning of any other category -> drop it *and* its indented continuation lines;
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

#: Absolute paths that would reveal the execution host. Kept separate from environment noise so a
#: caller can ask "does this text contain a host path?" without also rejecting an honest
#: `export LOKY_MAX_CPU_COUNT=4` in a shell script.
HOST_PATH = re.compile(r"/fh/|/home/|/Users/|/tmp/|site-packages")

#: Lines that reveal the execution host, or are pure environment noise.
LEAK = re.compile(
    HOST_PATH.pattern + r"|LOKY_|JAX_PLATFORMS|OMP_NUM_THREADS"
    r"|Kernel is running over TCP|IPKernelApp"
)

#: Warnings worth publishing are identified by **category**, never by the file CPython attributed
#: them to. `warnings.warn(..., stacklevel=2)` puts the *caller's* filename in the header, so a
#: path-based whitelist drops our own warning whenever a user raises it from a notebook cell -- and
#: prints "warnings preserved" while doing so. Category survives any stacklevel.
OUR_CATEGORY = re.compile(r"^(SpatialSmooth|Truncation)\w*Warning$")


def is_ours(category: str) -> bool:
    """Is this warning category one of ours? The **keep** rule."""
    return bool(OUR_CATEGORY.match(category))

#: A continuation of the preceding warning: CPython indents the offending source line.
CONTINUATION = re.compile(r"^\s+\S")


def scrub_text(text: str, stats: dict | None = None) -> str:
    """Return ``text`` with host paths removed and this package's warnings preserved.

    ``stats`` (optional) accumulates ``kept`` / ``dropped`` counts of *our* warnings, so the caller
    can report what happened rather than assert what it hoped happened.
    """
    out: list[str] = []
    dropping_continuation = False

    for line in text.splitlines():
        header = WARNING_HEADER.match(line)
        if header:
            if is_ours(header["category"]):
                # Keep the warning; redact only the path that precedes it. The module name is
                # whatever CPython attributed it to -- keep its basename, it is still useful.
                module = header["module"].replace("\\", "/").rsplit("/", 1)[-1]
                out.append(
                    f"{module}:{header['lineno']}: {header['category']}: {header['message']}"
                )
                if stats is not None:
                    stats["kept"] = stats.get("kept", 0) + 1
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


def _our_warning_lines(text: str) -> int:
    """Count our warnings by **category**, independent of the keep rule.

    Deliberately does not call :func:`is_ours`: if the keep rule regresses (as it did, twice), the
    loss detector must still see what went in. A detector that shares the bug it is watching for
    cannot see it -- which is how the last one blinded itself.
    """
    return sum(
        bool(m and OUR_CATEGORY.match(m["category"]))
        for m in (WARNING_HEADER.match(line) for line in text.splitlines())
    )


def scrub(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    kept = dropped = 0
    ours_before = ours_after = 0

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                original = out.get("text", "")
                ours_before += _our_warning_lines(original)
                cleaned = scrub_text(original)
                ours_after += _our_warning_lines(cleaned)
                if not cleaned.strip():
                    dropped += 1
                    continue
                out["text"] = cleaned + "\n"
                kept += 1
            outputs.append(out)
        cell["outputs"] = outputs

    nbformat.write(nb, path)
    # Report what happened; do not assert a property. An earlier version printed
    # "warnings preserved" unconditionally, while deleting the warning.
    lost = ours_before - ours_after
    print(
        f"stderr streams: {dropped} dropped, {kept} kept; "
        f"package warnings: {ours_before} seen, {ours_after} retained"
    )
    if lost > 0:
        print(
            f"ERROR: the scrub destroyed {lost} of this package's own warning(s). "
            "That is the failure this scrubber exists to prevent.",
            file=sys.stderr,
        )
        return 1

    # Fail loudly rather than publish a leak. Scan EVERY surface a path can reach the page
    # through -- stream text, execute_result / display_data text/plain and text/html, traceback
    # frames, and the cell source itself -- not just `stream`, which an earlier gate assumed.
    def _texts(cell):
        yield "".join(cell.get("source", []))
        for out in cell.get("outputs", []):
            yield "".join(out.get("text", []))
            yield "\n".join(out.get("traceback", []) or [])
            data = out.get("data") or {}
            for mime in ("text/plain", "text/html", "text/markdown"):
                yield "".join(data.get(mime, []))

    survivors = [
        line
        for cell in nb.cells
        for blob in _texts(cell)
        for line in blob.splitlines()
        if LEAK.search(line)
    ]
    if survivors:
        print(f"ERROR: {len(survivors)} leak line(s) survived: {survivors[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(scrub(sys.argv[1] if len(sys.argv) > 1 else "tutorial.ipynb"))
