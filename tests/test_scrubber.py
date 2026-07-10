"""The notebook scrubber must remove host paths without destroying our warnings.

Three ways this has been got wrong, each shipped, each caught by a skeptic and not by a test:

1. dropping every stderr stream deleted the truncation warning wholesale;
2. dropping every line matching a host path deleted it again -- CPython prints
   ``<path>:<lineno>: <Category>: <message>`` on **one line**, so the message rides on the line
   that carries the path;
3. whitelisting by the header's *filename* deleted it a third time, because
   ``warnings.warn(..., stacklevel=2)`` attributes the header to the **caller** -- so a user
   calling the low-level exports from a notebook cell lost the disclosure, while the scrubber
   printed "warnings preserved".

The principal fixture is **captured from a live interpreter**, not hand-written: an approximation
of CPython's format is one more sentence asserting a property nothing established.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))

scrub_notebook = pytest.importorskip("scrub_notebook")


def _capture_real_stderr(snippet: str, tmp_path) -> str:
    """Run ``snippet`` from a real ``.py`` file and return exactly what CPython wrote to stderr.

    A file, not ``python -c``: the latter reports its header as ``<string>:5:``, which is not the
    format a notebook kernel produces (``/tmp/ipykernel_.../1234.py:22:``). Testing against the
    wrong format is how the last fixture went stale.
    """
    script = tmp_path / "raise_warning.py"
    script.write_text(textwrap.dedent(snippet))
    proc = subprocess.run(
        [sys.executable, "-W", "always", str(script)],
        cwd=str(REPO),
        env={
            "PYTHONPATH": str(REPO / "src"),
            "MPLBACKEND": "Agg",
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stderr


@pytest.fixture(scope="module")
def real_warning_stderr(tmp_path_factory) -> str:
    """Verbatim stderr for our truncation warning, raised through the low-level export.

    ``stacklevel=2`` attributes the header to the *caller* -- this snippet -- not to
    ``spatial_smooth``. Exactly the case a filename-based whitelist silently dropped.
    """
    return _capture_real_stderr(
        """
        import numpy as np
        import spatial_smooth as ss
        coords = np.random.default_rng(0).random((1200, 2)) * 100.0
        ss.knn_gaussian_operator(coords, k=8)
        """,
        tmp_path_factory.mktemp("warn"),
    )


def test_the_captured_stderr_is_what_we_think_it_is(real_warning_stderr):
    """Guard the guard: if CPython's format changes, this fails before the tests built on it."""
    assert "TruncationWarning" in real_warning_stderr
    assert "truncates the kernel" in real_warning_stderr
    assert re.search(r"[\w./-]+\.py:\d+: \w*Warning: ", real_warning_stderr), (
        "expected a `path:lineno: Category: message` header"
    )


def test_our_warning_survives_even_when_attributed_to_the_caller(real_warning_stderr):
    """The G2 case: `stacklevel=2` names the caller's file, which is not in this package."""
    out = scrub_notebook.scrub_text(real_warning_stderr)
    assert "truncates the kernel" in out, "the warning text must survive the scrub"
    assert "effective bandwidth" in out, "the actionable half of the message must survive"
    assert "TruncationWarning" in out, "keep the category -- it is how the warning is identified"


def test_no_absolute_path_survives(real_warning_stderr):
    out = scrub_notebook.scrub_text(real_warning_stderr)
    for leaked in ("/fh/", "/home/", "/tmp/", "site-packages"):
        assert leaked not in out, f"{leaked!r} reached the public artifact"


def test_a_warning_raised_from_a_notebook_cell_survives():
    """In a kernel the caller is ``/tmp/ipykernel_.../<digits>.py`` -- a path, and not ours."""
    stderr = (
        "/tmp/ipykernel_31337/1841002313.py:22: TruncationWarning: KnnGaussian(k=8) truncates "
        "the kernel: only 14% of the Gaussian mass falls within each point's radius.\n"
        "  ss.knn_gaussian_operator(coords, k=8)\n"
    )
    out = scrub_notebook.scrub_text(stderr)
    assert "truncates the kernel" in out
    assert "/tmp/" not in out


def test_third_party_warning_is_dropped_with_its_continuation():
    stderr = (
        "/home/analyst/venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: "
        "IProgress not found. Please update jupyter and ipywidgets.\n"
        "  from .autonotebook import tqdm as notebook_tqdm\n"
    )
    out = scrub_notebook.scrub_text(stderr)
    assert out.strip() == "", "third-party chatter and its source echo both go"
    assert "autonotebook" not in out, "no decapitated fragment may remain"


def test_mixed_stream_keeps_only_ours(real_warning_stderr):
    third = (
        "/home/analyst/venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: x\n"
        "  from .autonotebook import tqdm as notebook_tqdm\n"
    )
    out = scrub_notebook.scrub_text(third + real_warning_stderr)
    assert "truncates the kernel" in out
    assert "autonotebook" not in out
    assert "/home/" not in out


def test_plain_leak_lines_are_dropped():
    out = scrub_notebook.scrub_text("LOKY_MAX_CPU_COUNT=2\nkeep me\n/home/someone/x.txt\n")
    assert out.splitlines() == ["keep me"]


def test_the_scrubber_has_a_committed_caller():
    """F3: the scrubber was orphaned -- its only caller was an untracked, host-specific script."""
    runner = REPO / "notebooks" / "run_tutorial.sh"
    assert runner.exists(), "a clean clone must be able to reproduce the published notebook"
    text = runner.read_text()
    assert "scrub_notebook.py" in text, "the runner must invoke the scrubber"
    assert "build_tutorial.py" in text
    # G6: forbid *any* absolute host path, not just the one I happened to think of. `LEAK` would
    # also reject the runner's honest `export LOKY_MAX_CPU_COUNT=4`, so match paths only.
    assert not scrub_notebook.HOST_PATH.search(text), "the runner must not hardcode host paths"


def _notebook_with(tmp_path, *, source="print(1)", outputs=()):
    import nbformat

    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(source)
    cell.outputs = list(outputs)
    nb.cells = [cell]
    path = tmp_path / "leaky.ipynb"
    nbformat.write(nb, str(path))
    return str(path)


@pytest.mark.parametrize(
    "surface",
    ["cell_source", "execute_result", "display_data", "traceback", "stream"],
    ids=lambda s: f"leak-in-{s}",
)
def test_leak_gate_fails_on_every_output_surface(tmp_path, surface):
    """G4: the gate scanned only `stream`. A path reaches the page through four other surfaces.

    Watched to fail: each of these exits 0 against the pre-G4 gate.
    """
    import nbformat

    # A deliberately fictional absolute path. `/fh/` is the prefix the gate must reject;
    # nothing after it names a real lab, user, or dataset.
    leak = "/fh/EXAMPLE/EXAMPLE_LAB/EXAMPLE_USER/data.h5ad"
    kwargs = {}
    if surface == "cell_source":
        kwargs["source"] = f"adata = read('{leak}')"
    elif surface == "execute_result":
        kwargs["outputs"] = [
            nbformat.v4.new_output("execute_result", data={"text/plain": f"PosixPath('{leak}')"},
                                   execution_count=1)
        ]
    elif surface == "display_data":
        kwargs["outputs"] = [
            nbformat.v4.new_output("display_data", data={"text/html": f"<pre>{leak}</pre>"})
        ]
    elif surface == "traceback":
        kwargs["outputs"] = [
            nbformat.v4.new_output("error", ename="OSError", evalue="no such file",
                                   traceback=[f'  File "{leak}", line 1', "OSError"])
        ]
    else:
        kwargs["outputs"] = [
            nbformat.v4.new_output("stream", name="stdout", text=f"wrote {leak}\n")
        ]

    path = _notebook_with(tmp_path, **kwargs)
    assert scrub_notebook.scrub(path) == 1, f"a leak in {surface} must fail the gate, not exit 0"


def test_leak_gate_passes_a_clean_notebook(tmp_path):
    """The gate must not cry wolf, or it will be switched off."""
    path = _notebook_with(tmp_path)
    assert scrub_notebook.scrub(path) == 0


def test_the_scrubber_fails_when_it_destroys_our_own_warning(tmp_path):
    """G2: if a package warning goes in and does not come out, that is an error, not a success."""
    import nbformat

    # Simulate the real regression: the KEEP rule stops recognising our warnings (as a
    # path-based whitelist did), while the loss DETECTOR is untouched. The gate must notice.
    original = scrub_notebook.is_ours
    scrub_notebook.is_ours = lambda category: False
    try:
        outputs = [
            nbformat.v4.new_output(
                "stream", name="stderr",
                text=(
                    "mod.py:1: TruncationWarning: truncates the kernel: only 14% ...\n"
                    "  ss.knn_gaussian_operator(coords, k=8)\n"
                ),
            )
        ]
        path = _notebook_with(tmp_path, outputs=outputs)
        assert scrub_notebook.scrub(path) == 1, (
            "destroying one of our own warnings must fail the run, not print 'preserved'"
        )
    finally:
        scrub_notebook.is_ours = original


def test_the_runner_guard_forbids_this_clusters_path(tmp_path):
    """G6: the guard forbade `/home/` but not `/fh/`. Watched to fail on a realistic runner."""
    bad = tmp_path / "bad_runner.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "ROOT=/fh/EXAMPLE/EXAMPLE_LAB/project\n"
        "export LOKY_MAX_CPU_COUNT=4\n"
    )
    assert scrub_notebook.HOST_PATH.search(bad.read_text()), (
        "a runner hardcoding a cluster path must be rejected by the guard"
    )
    # ... and an honest env var must NOT be rejected.
    ok = tmp_path / "ok_runner.sh"
    ok.write_text(
        "#!/bin/bash\n"
        "export LOKY_MAX_CPU_COUNT=4\n"
        "exec python scrub_notebook.py x\n"
    )
    assert not scrub_notebook.HOST_PATH.search(ok.read_text())
