"""Every install command we publish must be one that actually works.

The README, the docs and the tutorial advertise ``pip install "spatial-smooth[all]"``. That is a
documented instruction asserting a state someone has to establish -- so it must stay true. While the
name was unpublished the command could not resolve, and the advertising files said so ("pending")
and installed from a direct git reference instead. The package is now published on PyPI, so the bare
requirement resolves and is the command we advertise; the "pending" caveat would now be the lie.

These tests pin the *shape* of what we publish, offline (no network, no PyPI round-trip):

* every advertised ``pip install "<requirement>"`` is a parseable PEP 508 requirement;
* every self-referential command keeps its ``[all]`` extras -- an install that succeeds while
  silently skipping the extras is the same lie in a new costume;
* no advertising file still tells the reader the release is "pending"; the swap to PyPI is done.
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

#: This distribution's own name, as it appears in a requirement.
SELF = "spatial-smooth"


def _install_commands(path: pathlib.Path):
    """Yield ``(line_number, requirement_string, line)`` for each advertised `pip install`."""
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
def test_every_advertised_self_command_keeps_the_extras(path):
    """An install that succeeds while silently skipping the extras is the same lie in a new costume.

    Every command we advertise for our *own* distribution -- whether the published requirement or a
    direct git reference for the development version -- must carry the ``[all]`` extras.
    """
    for number, requirement, _line in _install_commands(path):
        parsed = Requirement(requirement)
        if parsed.name != SELF:
            continue  # e.g. `pip install "kompot>=0.7.0"` -- a dependency, not us
        assert parsed.extras == {"all"}, (
            f"{path.name}:{number} advertises `pip install \"{requirement}\"`, which drops the "
            f"extras: it installs {SELF} without any optional backend. Keep the `[all]` marker."
        )


def test_the_primary_install_command_is_the_published_pypi_form():
    """`spatial-smooth` is on PyPI, so the command we lead with is the bare, resolvable one.

    Not a git reference (that is the development-version fallback), and with its ``[all]`` extras
    intact. This is the command a reader copies first, so it is the one that must resolve today.
    """
    readme = (REPO / "README.md").read_text()
    primary = next(
        requirement
        for requirement in PIP_INSTALL.findall(readme)
        if Requirement(requirement).name == SELF
    )
    parsed = Requirement(primary)
    assert parsed.url is None, (
        f"the primary README command is a direct reference ({primary}); once published it should "
        "lead with the bare `spatial-smooth[all]` and keep the git form only as a dev fallback"
    )
    assert parsed.extras == {"all"}, f"the advertised command lost its extras: {primary}"


@pytest.mark.parametrize("path", ADVERTISING, ids=lambda p: p.name)
def test_no_advertising_file_still_calls_the_release_pending(path):
    """The swap to PyPI is done: "pending" was true before publication and is now the lie."""
    assert "pending" not in path.read_text().lower(), (
        f"{path.name} still calls the PyPI release 'pending', but {SELF} is published -- "
        "the bare `pip install \"spatial-smooth[all]\"` resolves now"
    )
