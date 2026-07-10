"""Every install command we publish must be one that actually works.

The README, the docs and the tutorial all advertised ``pip install "spatial-smooth[all]"`` while
the name was unpublished on PyPI, so every reader who followed our own instructions got
``no matching distribution``. That is a documented instruction asserting a state nobody
established -- the same defect this package has produced repeatedly, aimed at the user.

Verified out-of-band by execution (``uv pip install --dry-run``), which cannot run in CI without
network and repository access:

* ``spatial-smooth[all]``                         -> "no versions of spatial-smooth[all]"; unsatisfiable
* ``spatial-smooth[all] @ git+https://…``         -> resolves, and pulls kompot, palantir, squidpy,
  scanpy and KDEpy; without the ``[all]`` marker it pulls **none** of them.

These tests pin the *shape* of what we publish, offline: a bare requirement on an unpublished name
is a promise we cannot keep.
"""
from __future__ import annotations

import pathlib
import re

import pytest

packaging_requirements = pytest.importorskip("packaging.requirements")
Requirement = packaging_requirements.Requirement
InvalidRequirement = packaging_requirements.InvalidRequirement

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Files that advertise an install command to a reader.
ADVERTISING = (
    REPO / "README.md",
    REPO / "docs" / "source" / "installation.rst",
    REPO / "notebooks" / "build_tutorial.py",
)

#: ``pip install "<requirement>"`` -- the form we publish.
PIP_INSTALL = re.compile(r'pip install\s+"([^"]+)"')

#: Prose that explicitly describes the *future* PyPI command rather than advertising it now.
FUTURE_TENSE = re.compile(r"(becomes|will be|once .*published)", re.IGNORECASE)

#: This distribution's own name, as it appears in a requirement.
SELF = "spatial-smooth"


def _install_commands(path: pathlib.Path):
    """Yield ``(line_number, requirement_string)`` for each advertised `pip install`."""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        for requirement in PIP_INSTALL.findall(line):
            yield number, requirement, line


@pytest.mark.parametrize("path", ADVERTISING, ids=lambda p: p.name)
def test_every_advertised_install_command_is_valid_pep508(path):
    for number, requirement, _line in _install_commands(path):
        try:
            Requirement(requirement)
        except InvalidRequirement as exc:  # pragma: no cover - only on a typo
            pytest.fail(f"{path.name}:{number} advertises an unparseable requirement: {exc}")


@pytest.mark.parametrize("path", ADVERTISING, ids=lambda p: p.name)
def test_we_never_advertise_an_unpublished_name_without_a_source(path):
    """Until `spatial-smooth` is on PyPI, a bare requirement on it cannot resolve.

    A direct reference (PEP 508 ``name[extras] @ url``) preserves the extras and does resolve. The
    only permitted bare mention is prose describing what the command *will become* once published.
    """
    for number, requirement, line in _install_commands(path):
        parsed = Requirement(requirement)
        if parsed.name != SELF:
            continue  # e.g. `pip install "kompot>=0.7.0"` -- kompot IS on PyPI
        if parsed.url:
            continue  # direct reference: resolvable today
        assert FUTURE_TENSE.search(line), (
            f"{path.name}:{number} tells the reader to run `pip install \"{requirement}\"`, "
            f"but {SELF!r} is not on PyPI, so it fails with 'no matching distribution'. "
            "Use a direct reference, or mark the line as describing the future release."
        )


def test_the_primary_install_command_keeps_the_extras():
    """An install that succeeds while silently skipping the extras is the same lie in a new costume."""
    readme = (REPO / "README.md").read_text()
    primary = next(
        requirement
        for requirement in PIP_INSTALL.findall(readme)
        if Requirement(requirement).name == SELF and Requirement(requirement).url
    )
    parsed = Requirement(primary)
    assert parsed.extras == {"all"}, f"the advertised command lost its extras: {primary}"
    assert parsed.url.startswith("git+https://"), "the source must be fetchable over https"


def test_the_swap_to_pypi_is_a_one_line_edit():
    """When the operator publishes, the git line becomes the fallback -- not a rewrite."""
    for path in (REPO / "README.md", REPO / "docs" / "source" / "installation.rst"):
        text = path.read_text()
        assert "pending" in text.lower(), f"{path.name} must say the PyPI release is pending"
        assert re.search(r'spatial-smooth\[all\]"', text), (
            f"{path.name} must show the future PyPI command so the swap is a one-line edit"
        )
