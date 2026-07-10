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


def _stream_text(notebook) -> str:
    return "\n".join(
        "".join(out.get("text", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )


def test_tutorial_leaks_no_host_paths(notebook):
    hits = LEAK.findall(NOTEBOOK.read_text())
    assert not hits, f"tutorial leaks host-specific strings into public docs: {set(hits)}"


def test_tutorial_is_executed_with_figures(notebook):
    if not _executed(notebook):
        pytest.skip("tutorial has not been executed in this checkout")
    source = NOTEBOOK.read_text()
    assert source.count('"image/png"') >= 5, "the tutorial should ship its rendered figures"
    assert '"output_type": "error"' not in source, "the tutorial contains an error output"


def test_tutorial_surfaces_the_truncation_warning(notebook):
    """R1: the level-three pipeline uses k=64 and must show, not hide, the warning that fires."""
    if not _executed(notebook):
        pytest.skip("tutorial has not been executed in this checkout")
    text = _stream_text(notebook)
    assert "truncates the kernel" in text, (
        "the tutorial's own pipeline trips the truncation warning; the notebook must display it "
        "rather than scrub it away"
    )
    assert "sigma_effective" in NOTEBOOK.read_text(), (
        "the notebook must tell the reader which bandwidth to quote"
    )


def test_tutorial_does_not_subsample_or_use_time_magics(notebook):
    source = NOTEBOOK.read_text()
    assert "%time" not in source, "%time is noise for the intended reader"
    assert "12_000" not in source, "the tutorial runs on the full section"


def test_tutorial_states_the_visualization_only_warning(notebook):
    source = NOTEBOOK.read_text().lower()
    assert "for looking, not for measuring" in source or "visualization only" in source
