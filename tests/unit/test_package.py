def test_version_is_exposed() -> None:
    import alchemypose

    assert isinstance(alchemypose.__version__, str)
    assert alchemypose.__version__.count(".") == 2


def test_importing_package_does_not_import_qt() -> None:
    """core/ must be usable on a machine with no display and no Qt."""
    import subprocess
    import sys

    code = "import alchemypose, sys; assert not [m for m in sys.modules if m.startswith('PySide6')]"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
