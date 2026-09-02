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


@app.command()
def gui() -> None:
    """Open the AlchemyPose window."""
    from alchemypose.gui.app import main

    raise typer.Exit(main())


@app.command()
def image(
    path: Path = typer.Argument(..., help="Image to analyse."),
    preset: str = typer.Option("performance", help="lite | balanced | performance"),
    json_out: Path | None = typer.Option(None, "--json", help="Write the result here instead of stdout."),
    csv_out: Path | None = typer.Option(None, "--csv", help="Also write a flat CSV here."),
) -> None:
    """Run pose estimation on one image and print the result as JSON."""
    if not path.exists():
        typer.echo(f"Error: {path} not found.", err=True)
        raise typer.Exit(1)

    from alchemypose.core.serialize import to_csv, to_json
    from alchemypose.inference.rtmw import RTMWEngine
    from alchemypose.media.image import load_image

    try:
        result = RTMWEngine(preset=preset).infer(load_image(path), source=str(path))
    except AlchemyPoseError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

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
def models_fetch(preset: str = typer.Option("performance")) -> None:
    """Download a preset's models now, rather than on first use."""
    from alchemypose.inference import weights

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
