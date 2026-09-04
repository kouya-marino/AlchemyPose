# src/alchemypose/cli.py
"""The command line. ``alchemypose`` is the app; the rest is for scripting."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from alchemypose import __version__
from alchemypose.errors import AlchemyPoseError
from alchemypose.inference import weights

if TYPE_CHECKING:  # imported for real only when a download actually starts
    from rich.progress import Progress, TaskID

app = typer.Typer(
    help="Whole-body human pose visualisation. With no command, opens the window.",
    # Deliberately *not* ``no_args_is_help=True``, and written out rather than
    # left to the default because the default is the whole mechanism. click
    # checks this flag in ``Group.parse_args``, which runs before
    # ``invoke_without_command`` is ever consulted, so with both set a bare
    # ``alchemypose`` prints the help and exits 2 and the callback below never
    # runs at all. Re-adding the flag would silently take the window away from
    # the bare command.
    no_args_is_help=False,
)
models_app = typer.Typer(help="Manage cached ONNX model files.")
app.add_typer(models_app, name="models")

#: Derived, never typed out. A hand-written copy is how ``--help`` ends up
#: advertising three presets while the rejection message names four.
_PRESET_HELP = " | ".join(weights.PRESETS)


def _stderr_is_a_terminal() -> bool:
    """Whether stderr is something a moving bar belongs on.

    Wrapped rather than inlined because every part of it can be untrue at once:
    ``sys.stderr`` is ``None`` under a windowed interpreter, closed by the time
    an atexit path runs, and a plain object with no ``isatty`` under some test
    harnesses. None of those are worth an exception on the way to deciding how
    to print a sentence.
    """
    stream = sys.stderr
    try:
        return stream is not None and bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class _DownloadFeedback:
    """Says what is being downloaded, on stderr, for as long as it takes.

    ``alchemypose image photo.jpg`` on a cold cache printed nothing at all
    until the result appeared, a few hundred megabytes and several minutes
    later — from the outside, indistinguishable from a command that had hung.
    This fills that silence, and is built to be invisible the rest of the time:

    * Nothing is constructed and nothing is printed until bytes actually start
      moving. ``weights.ensure`` calls back only once it has decided to fetch,
      so a warm cache — the common case, and the one that has to stay fast and
      quiet — gets exactly the output it always had: none.
    * Everything goes to **stderr**, so ``alchemypose image photo.jpg > out.json``
      still puts nothing but JSON in the file.
    * The bar is drawn only when stderr is a terminal. Piped or redirected, the
      same facts arrive as one plain sentence per model with not one escape
      sequence in it, because the other end of that pipe is usually a log.
      rich's own handling of a non-terminal console is no substitute, and was
      measured rather than assumed: it renders the bar once, in box-drawing
      characters, when the transfer is already *over* — the one moment at which
      the number has stopped being any use.
    """

    def __init__(self) -> None:
        self._interactive = _stderr_is_a_terminal()
        self._announced: set[str] = set()
        self._said_where = False
        self._bars: Progress | None = None
        self._tasks: dict[str, TaskID] = {}

    def for_model(self, key: str) -> Callable[[int, int], None]:
        """The ``progress=`` callable to hand ``weights.ensure(key)``.

        One per model, because the key is the half of the message ``ensure``
        does not pass on — the callback is told bytes, not which file they
        belong to.
        """

        def report(done: int, total: int) -> None:
            if key not in self._announced:
                self._announce(key, total)
            self._advance(key, done)

        return report

    def close(self) -> None:
        """Give the terminal back. A no-op when nothing was ever drawn.

        Called between models as well as at the end: a live region and any
        other writer sharing one terminal is how a line ends up half
        overwritten, and ``models fetch`` prints between downloads.
        """
        bars, self._bars = self._bars, None
        self._tasks.clear()  # a TaskID belongs to the Progress that issued it
        if bars is not None:
            bars.stop()

    # ----------------------------------------------------------------- detail

    def _announce(self, key: str, total: int) -> None:
        """The one line that has to arrive *before* the wait, not after it."""
        self._announced.add(key)
        if not self._said_where:
            # Where they land, and that this is not a per-run cost, are the two
            # questions the first sight of a 213 MB download raises. Once.
            self._said_where = True
            typer.echo(f"Model weights are downloaded once and kept in {weights.cache_dir()}.", err=True)

        headline = f"Downloading {key} ({weights.human_size(total)})"
        if not self._interactive:
            typer.echo(f"{headline} ...", err=True)
            return
        self._tasks[key] = self._open_bars().add_task(headline, total=total or None)

    def _advance(self, key: str, done: int) -> None:
        task = self._tasks.get(key)
        if task is not None and self._bars is not None:
            self._bars.update(task, completed=done)

    def _open_bars(self) -> Progress:
        """The rich import happens here, and so only on a cold cache.

        Every other invocation — ``--version``, a warm ``image``, a rejected
        preset — never touches this method, and so never pays for importing
        rich to decide it had nothing to draw.
        """
        if self._bars is None:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )

            self._bars = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                # A Console of its own, never ``rich.get_console()``: that one
                # is a process-wide singleton pointed at stdout, and stdout is
                # where the JSON goes. ``stderr=True`` re-reads ``sys.stderr``
                # on every write rather than capturing it now, which is what a
                # harness that swaps the stream out from under the command
                # needs.
                console=Console(stderr=True),
            )
            self._bars.start()
        return self._bars


@contextmanager
def _download_feedback() -> Iterator[_DownloadFeedback]:
    """Scope the terminal hand-over: however this ends, the bar is closed."""
    feedback = _DownloadFeedback()
    try:
        yield feedback
    finally:
        feedback.close()


def _fetch_weights(preset: str) -> None:
    """Put the preset's models on disk before anything slow and silent wants them.

    ``RTMWEngine`` calls ``weights.ensure`` itself and would do it without a
    word; fetching here first means the download happens in the one place that
    knows whether a terminal is watching. A model already cached costs one
    checksum and prints nothing, and the engine's own ``ensure`` a moment later
    costs a second one — a tenth of a second across the largest preset, against
    a download this exists to narrate.
    """
    with _download_feedback() as feedback:
        for key in weights.PRESETS[preset]:
            weights.ensure(key, progress=feedback.for_model(key))
            feedback.close()


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
    ``_PRESET_HELP`` is built from the same dict, so ``--help`` and this message
    cannot come to name different sets.
    """
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


def _version_and_exit(ctx: typer.Context, value: bool) -> bool:
    """Answer ``--version`` during parsing, before any command can run.

    ``is_eager`` is what makes it "during parsing": the version is printed and
    the process stops before the callback below gets as far as deciding whether
    to open a window, which matters now that deciding nothing is what opens one.

    ``resilient_parsing`` is set while click is computing a shell completion.
    Writing to stdout in that pass would put the version string into the list of
    completions the shell is reading.
    """
    if value and not ctx.resilient_parsing:
        typer.echo(__version__)
        raise typer.Exit()
    return value


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the version and exit.",
        is_eager=True,
        callback=_version_and_exit,
    ),
) -> None:
    """Open the window when no subcommand is given.

    ``alchemypose`` on its own is what a person types straight after
    ``pip install alchemypose``, and a page of help is not what they came for.
    ``invoke_without_command`` is what lets this run with no subcommand at all;
    the guard below is what stops it running *ahead* of one, because click
    invokes a group callback for every subcommand too. Without the guard,
    ``alchemypose image photo.jpg`` opens a window and never analyses the
    photograph at all — ``gui()`` exits with the window's status, and click
    never reaches the subcommand it already resolved. Measured on exactly that
    edit: exit code 43 from a stubbed window, and no output whatsoever.

    A mistyped subcommand does not reach here either, and that is click's doing
    rather than this function's: ``Group.invoke`` resolves the name and fails on
    an unknown one *before* it invokes the group callback, so
    ``alchemypose imagee`` exits 2 with a usage message instead of opening a
    window over a typo. ``tests/unit/test_cli.py`` pins that, because it is the
    part of this arrangement that a later refactor could quietly lose.

    ``gui()`` is called rather than reimplemented so the bare command and
    ``alchemypose gui`` cannot come to mean different things.
    """
    if ctx.invoked_subcommand is not None:
        return
    gui()


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
        _fetch_weights(preset)
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
    preset = _checked_preset(preset)
    with _download_feedback() as feedback:
        for key in weights.PRESETS[preset]:
            # Named on stdout whether or not it turns out to need fetching —
            # this is the command whose whole subject is the cache, so "nothing
            # to do for this one" is an answer, not noise. The size and the bar
            # come from the feedback on stderr, and only if bytes move.
            typer.echo(f"Fetching {key} ...")
            weights.ensure(key, progress=feedback.for_model(key))
            feedback.close()
    typer.echo("Done.")


@models_app.command("clear")
def models_clear() -> None:
    """Delete every cached model file."""
    removed = 0
    for file in weights.cache_dir().glob("*.onnx"):
        file.unlink()
        removed += 1
    typer.echo(f"Removed {removed} file(s).")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)
