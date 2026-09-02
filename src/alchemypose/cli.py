# src/alchemypose/cli.py
"""The command line. ``alchemypose gui`` is the app; the rest is for scripting."""

from __future__ import annotations

from pathlib import Path

import typer

from alchemypose import __version__
from alchemypose.errors import AlchemyPoseError

app = typer.Typer(help="Whole-body human pose visualisation.", no_args_is_help=True)
models_app = typer.Typer(help="Manage cached ONNX model files.")
app.add_typer(models_app, name="models")

_PRESET_HELP = "lite | balanced | performance"


def _fail(message: str) -> typer.Exit:
    """Print a one-line error and build the exit to raise.

    A mistyped flag is a user error, not a defect, and must never reach the
    terminal as a traceback.
    """
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(1)


def _checked_preset(preset: str) -> str:
    """Return ``preset`` if it names one, else exit with the valid choices.

    ``weights.PRESETS`` is the single source of truth, so a preset added there
    needs no second edit here — which an ``Enum`` on the option would have.
    """
    from alchemypose.inference import weights

    if preset not in weights.PRESETS:
        raise _fail(f"unknown preset {preset!r}. Choose one of: {', '.join(weights.PRESETS)}.")
    return preset


def _checked_output(path: Path | None, flag: str) -> Path | None:
    """Reject an output path whose directory is missing, before anything runs.

    Inference costs seconds and a model download costs minutes; discovering the
    destination does not exist only once there is a result to write throws all
    of that away.
    """
    if path is not None and not path.parent.is_dir():
        raise _fail(f"{flag} directory {path.parent} does not exist.")
    return path


@app.command()
def gui() -> None:
    """Open the AlchemyPose window."""
    from alchemypose.gui.app import main

    raise typer.Exit(main())


@app.command()
def image(
    path: Path = typer.Argument(..., help="Image to analyse."),
    preset: str = typer.Option("performance", help=_PRESET_HELP),
    json_out: Path | None = typer.Option(None, "--json", help="Write the result here instead of stdout."),
    csv_out: Path | None = typer.Option(None, "--csv", help="Also write a flat CSV here."),
) -> None:
    """Run pose estimation on one image and print the result as JSON."""
    preset = _checked_preset(preset)
    if not path.exists():
        raise _fail(f"{path} not found.")
    json_out = _checked_output(json_out, "--json")
    csv_out = _checked_output(csv_out, "--csv")

    from alchemypose.core.serialize import to_csv, to_json
    from alchemypose.inference.rtmw import RTMWEngine
    from alchemypose.media.image import load_image

    try:
        result = RTMWEngine(preset=preset).infer(load_image(path), source=str(path))
    except AlchemyPoseError as exc:
        raise _fail(str(exc)) from exc

    payload = to_json(result)
    if json_out is not None:
        json_out.write_text(payload)
        typer.echo(f"Wrote {json_out}")
    else:
        typer.echo(payload)
    if csv_out is not None:
        csv_out.write_text(to_csv(result))
        typer.echo(f"Wrote {csv_out}")


@models_app.command("list")
def models_list() -> None:
    """Show the presets, their models, and whether they are cached."""
    from alchemypose.inference import weights

    typer.echo(f"Cache: {weights.cache_dir()}")
    for preset, keys in weights.PRESETS.items():
        typer.echo(f"\n{preset}:")
        for key in keys:
            spec = weights.REGISTRY[key]
            state = "cached" if (weights.cache_dir() / spec.filename).exists() else "not downloaded"
            typer.echo(f"  {key:<22} {state}")


@models_app.command("fetch")
def models_fetch(preset: str = typer.Option("performance", help=_PRESET_HELP)) -> None:
    """Download a preset's models now, rather than on first use."""
    from alchemypose.inference import weights

    preset = _checked_preset(preset)
    for key in weights.PRESETS[preset]:
        typer.echo(f"Fetching {key} ...")
        weights.ensure(key)
    typer.echo("Done.")


@models_app.command("clear")
def models_clear() -> None:
    """Delete every cached model file."""
    from alchemypose.inference import weights

    removed = 0
    for file in weights.cache_dir().glob("*.onnx"):
        file.unlink()
        removed += 1
    typer.echo(f"Removed {removed} file(s).")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)
