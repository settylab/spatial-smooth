"""The notebook scrubber must remove host paths without decapitating warnings.

CPython prints a warning as ``<path>/steps.py:202: UserWarning: <message>`` -- the message rides
on the same line as the path. A line-based path filter therefore deletes the warning text and
leaves its orphaned source echo behind. This was shipped, and only a skeptic's probe caught it:
the tutorial appeared to display the warning solely because a notebook cell printed it to *stdout*.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))

scrub_notebook = pytest.importorskip("scrub_notebook")

#: Verbatim stderr for the warning the tutorial's own pipeline raises.
OUR_WARNING = (
    "/home/analyst/projects/spatial-smooth/src/spatial_smooth/steps.py:202: "
    "UserWarning: KnnGaussian(k=64) truncates the kernel: only 69% of the Gaussian mass falls "
    "within each cell's 64-neighbour radius, so the effective bandwidth is 35.7 (nominal sigma "
    "52.1) and varies with local density. Raise k, or quote provenance()'s 'sigma_effective'.\n"
    "  warnings.warn(\n"
)

THIRD_PARTY = (
    "/home/analyst/venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: "
    "IProgress not found. Please update jupyter and ipywidgets.\n"
    "  from .autonotebook import tqdm as notebook_tqdm\n"
)


def test_our_warning_survives_with_the_path_redacted():
    out = scrub_notebook.scrub_text(OUR_WARNING)
    assert "truncates the kernel" in out, "the warning text must survive the scrub"
    assert "sigma_effective" in out, "the actionable half of the message must survive"
    assert "/home/" not in out, "no absolute path may reach public docs"
    assert "spatial_smooth/steps.py:202" in out, "keep a useful, redacted provenance"


def test_third_party_warning_is_dropped_with_its_continuation():
    out = scrub_notebook.scrub_text(THIRD_PARTY)
    assert out.strip() == "", "third-party chatter and its source echo both go"
    assert "autonotebook" not in out, "no decapitated fragment may remain"


def test_mixed_stream_keeps_only_ours():
    out = scrub_notebook.scrub_text(THIRD_PARTY + OUR_WARNING)
    assert "truncates the kernel" in out
    assert "IProgress" not in out
    assert "from .autonotebook" not in out
    assert "/home/" not in out


def test_plain_leak_lines_are_dropped():
    out = scrub_notebook.scrub_text("LOKY_MAX_CPU_COUNT=2\nkeep me\n/home/someone/x.txt\n")
    assert out.splitlines() == ["keep me"]


def test_the_scrubber_has_a_committed_caller():
    """F3: the scrubber was orphaned -- its only caller was an untracked, host-specific sbatch."""
    root = pathlib.Path(__file__).resolve().parents[1]
    runner = root / "notebooks" / "run_tutorial.sh"
    assert runner.exists(), "a clean clone must be able to reproduce the published notebook"
    text = runner.read_text()
    assert "scrub_notebook.py" in text, "the runner must invoke the scrubber"
    assert "build_tutorial.py" in text
    assert "/home/" not in text, "the runner must not hardcode host paths"
