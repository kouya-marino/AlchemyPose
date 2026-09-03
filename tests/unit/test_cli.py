import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from alchemypose.cli import app

runner = CliRunner()


def test_models_list_names_every_preset() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    for preset in ("lite", "balanced", "performance"):
        assert preset in result.stdout


def test_image_reports_a_missing_file_without_a_traceback() -> None:
    result = runner.invoke(app, ["image", "/definitely/not/here.jpg"])
    assert result.exit_code != 0
    # The error goes to stderr (standard CLI practice); `.output` is the combined
    # stream, i.e. what a user actually sees in a terminal.
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


def test_image_rejects_an_unknown_preset_without_a_traceback() -> None:
    result = runner.invoke(app, ["image", "/definitely/not/here.jpg", "--preset", "nope"])
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "unknown preset" in result.output.lower()
    # The message has to name the choices; "that is wrong" without "try these"
    # sends the user back to --help for something the CLI already knows.
    for preset in ("lite", "balanced", "performance"):
        assert preset in result.output


def test_models_fetch_rejects_an_unknown_preset_without_a_traceback() -> None:
    result = runner.invoke(app, ["models", "fetch", "--preset", "nope"])
    assert result.exit_code != 0
    # A KeyError escaping the command is what a raw traceback looks like in a
    # real terminal; CliRunner swallows it into `.exception` instead of stderr.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "unknown preset" in result.output.lower()
    for preset in ("lite", "balanced", "performance"):
        assert preset in result.output


@pytest.mark.parametrize("flag", ["--json", "--csv"])
def test_image_rejects_a_missing_output_directory_before_loading_a_model(
    flag: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination is checked up front, not after the inference is paid for.

    Building an engine downloads and compiles hundreds of megabytes; finding out
    only at write time that the directory does not exist throws that away and
    leaves the user with a traceback instead of a result.
    """
    from alchemypose.inference import rtmw

    def _too_late(*args: object, **kwargs: object) -> None:
        raise AssertionError("the output path must be checked before a model is loaded")

    monkeypatch.setattr(rtmw, "RTMWEngine", _too_late)

    source = tmp_path / "frame.png"
    source.write_bytes(b"")
    result = runner.invoke(app, ["image", str(source), flag, str(tmp_path / "missing" / "out")])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "does not exist" in result.output
    assert str(tmp_path / "missing") in result.output


def test_help_names_a_preset_added_to_the_registry() -> None:
    """``--help`` reads the preset list off ``weights.PRESETS`` instead of repeating it.

    A second, hand-maintained copy drifts the moment a preset is added: the new
    name is accepted and named in the rejection message, yet ``--help`` still
    advertises the old three, so the one place a user looks to find the choices
    is the one place that is wrong.

    The preset has to be added before ``alchemypose.cli`` is imported, because
    Typer bakes the option's help text into the command when the module is
    executed — hence the subprocess rather than a monkeypatch. It also keeps the
    registry edit out of this process, where it would leak into every test that
    runs after this one.
    """
    code = (
        "from alchemypose.inference import weights\n"
        "weights.PRESETS['quantum'] = ('yolox_nano', 'rtmw_m_256x192')\n"
        "from typer.testing import CliRunner\n"
        "from alchemypose.cli import app\n"
        "runner = CliRunner()\n"
        "for argv in (['image', '--help'], ['models', 'fetch', '--help']):\n"
        "    result = runner.invoke(app, argv)\n"
        "    assert result.exit_code == 0, result.output\n"
        "    print(result.output)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
    )

    assert proc.returncode == 0, proc.stderr
    assert "quantum" in proc.stdout, proc.stdout
    for preset in ("lite", "balanced", "performance"):
        assert preset in proc.stdout
