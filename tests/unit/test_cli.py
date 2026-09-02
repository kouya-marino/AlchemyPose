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
