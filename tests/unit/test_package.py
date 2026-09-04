import importlib.metadata

import pytest


def test_version_is_exposed() -> None:
    import alchemypose

    assert isinstance(alchemypose.__version__, str)
    assert alchemypose.__version__.count(".") == 2


def test_the_installed_metadata_agrees_with_the_attribute() -> None:
    """One version, written once, reaching both places it is read from.

    ``alchemypose.__version__`` is the single source: ``pyproject.toml``
    declares ``dynamic = ["version"]`` and setuptools reads the attribute out of
    the source with ast at build time. Nothing keeps the two in step by itself,
    though — the wiring is one line of TOML, and deleting it or mistyping the
    attribute path puts a hard-coded number back in the metadata that nothing
    else in the suite would look at. Then ``pip show alchemypose`` and
    ``alchemypose --version`` say different things, and the number a user quotes
    when something goes wrong is the wrong one.

    What this actually compares is the *installed* distribution against the
    *imported* module, so it also catches the ordinary version of the same
    mistake: a bumped attribute in a working tree that was never reinstalled, so
    the metadata still describes the previous release.

    Where the package is not installed at all — a checkout run with
    ``PYTHONPATH=src`` and nothing else, which is how a packager or a CI job
    that skips the install step sees it — there is no metadata to disagree with
    and the test skips rather than failing. The skip is the honest outcome: the
    invariant is about an installed copy and there is no installed copy. It does
    mean this guard is only load-bearing where ``pip install`` has run, which is
    every environment that could publish or consume a release.
    """
    import alchemypose

    try:
        installed = importlib.metadata.version("alchemypose")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("alchemypose is not installed in this environment; there is no metadata to compare against")

    assert installed == alchemypose.__version__, (
        f"the installed distribution says {installed!r} and alchemypose.__version__ says "
        f"{alchemypose.__version__!r}. pyproject.toml reads the version from that attribute, so either "
        "[tool.setuptools.dynamic] has stopped pointing at it or this environment is running a stale "
        "install of an older version — reinstall and check the metadata before releasing anything."
    )


def test_importing_package_does_not_import_qt() -> None:
    """core/ must be usable on a machine with no display and no Qt."""
    import subprocess
    import sys

    code = "import alchemypose, sys; assert not [m for m in sys.modules if m.startswith('PySide6')]"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
