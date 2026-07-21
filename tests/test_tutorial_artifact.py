"""The published tutorial is an artifact with contracts of its own.

It renders onto a public documentation site, so it must not leak the execution host. And because
its level-three example deliberately trips the kernel-truncation warning, that warning must
survive whatever scrubbing the execution pipeline does -- an earlier scrubber dropped every
stderr stream and silently deleted the one disclosure the section exists to teach.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "tutorial.ipynb"

#: Anything that reveals where the notebook was executed.
LEAK = re.compile(r"/fh/|/home/|site-packages|LOKY_|JAX_PLATFORMS|Kernel is running over TCP")


@pytest.fixture(scope="module")
def notebook():
    if not NOTEBOOK.exists():  # pragma: no cover - the repo always ships it
        pytest.skip(f"{NOTEBOOK} not present")
    return json.loads(NOTEBOOK.read_text())


def _executed(notebook) -> bool:
    return any(
        cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def _stream_text(notebook, name: str) -> str:
    """Concatenate one stream by **name**.

    G1: this used to merge stdout and stderr. That is precisely why the decapitated-warning bug
    survived a round: a stdout `print` of the message satisfied the assertion while the stderr
    stream had been gutted. A detector that cannot distinguish the two cannot catch it.
    """
    return "\n".join(
        "".join(out.get("text", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream" and out.get("name") == name
    )


def _readable_text(notebook) -> str:
    """Every human-readable surface a host path could reach the rendered page through.

    Cell source plus the stream, ``text/plain`` / ``text/html`` / ``text/markdown`` and traceback
    outputs -- but **not** base64 media (``image/png`` and friends). Encoded image bytes are not
    readable and are not a leak vector, yet a large enough blob contains ``/fh/`` or ``/home/`` by
    chance -- so scanning the raw file turned an honest figure into a spurious failure. This mirrors
    exactly what ``notebooks/scrub_notebook.py`` cleans, so the two agree on what "a leak" means.
    """
    parts: list[str] = []
    for cell in notebook["cells"]:
        parts.append("".join(cell.get("source", [])))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", [])))
            parts.append("\n".join(out.get("traceback", []) or []))
            data = out.get("data") or {}
            for mime in ("text/plain", "text/html", "text/markdown"):
                parts.append("".join(data.get(mime, [])))
    return "\n".join(parts)


def test_tutorial_leaks_no_host_paths(notebook):
    hits = LEAK.findall(_readable_text(notebook))
    assert not hits, f"tutorial leaks host-specific strings into public docs: {set(hits)}"


def test_tutorial_is_executed_with_figures(notebook):
    if not _executed(notebook):
        pytest.skip("tutorial has not been executed in this checkout")
    source = NOTEBOOK.read_text()
    assert source.count('"image/png"') >= 5, "the tutorial should ship its rendered figures"
    assert '"output_type": "error"' not in source, "the tutorial contains an error output"


WARNING_HEADER = re.compile(
    r"^[\w./-]+\.py:\d+: \w*Warning: .*truncates the kernel", re.MULTILINE
)
ORPHAN_ECHO = re.compile(r"^\s+\S")


def test_tutorial_surfaces_the_truncation_warning_on_stderr(notebook):
    """R1/G1: the level-three pipeline trips the warning; a real warning header must survive.

    Asserted against **stderr**, and against a header, not a bare substring. The previous version
    passed on a notebook whose stderr had been decapitated, because a stdout copy carried the text.
    """
    if not _executed(notebook):
        pytest.skip("tutorial has not been executed in this checkout")
    stderr = _stream_text(notebook, "stderr")
    assert WARNING_HEADER.search(stderr), (
        "the published notebook must carry a genuine warning header on stderr "
        f"(got: {stderr[:200]!r})"
    )


def test_tutorial_has_no_orphaned_warning_fragments(notebook):
    """G1: an indented source echo with no header above it is a decapitated warning."""
    if not _executed(notebook):
        pytest.skip("tutorial has not been executed in this checkout")
    stderr = _stream_text(notebook, "stderr")
    header_seen = False
    for line in stderr.splitlines():
        if re.match(r"^[\w./-]+\.py:\d+: \w*Warning: ", line):
            header_seen = True
            continue
        if ORPHAN_ECHO.match(line):
            assert header_seen, f"orphaned warning fragment with no header: {line!r}"
        elif line.strip():
            header_seen = False


def test_tutorial_tells_the_reader_which_bandwidth_to_quote(notebook):
    assert "sigma_effective" in NOTEBOOK.read_text()


def test_tutorial_does_not_subsample_or_use_time_magics(notebook):
    source = NOTEBOOK.read_text()
    assert "%time" not in source, "%time is noise for the intended reader"
    assert "12_000" not in source, "the tutorial runs on the full section"


def test_tutorial_states_the_visualization_only_warning(notebook):
    source = NOTEBOOK.read_text().lower()
    assert "for looking, not for measuring" in source or "visualization only" in source
