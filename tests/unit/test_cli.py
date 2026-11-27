import hashlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from alchemypose import __version__
from alchemypose.cli import app

runner = CliRunner()

#: How many bytes of stand-in each model in ``cold_cache`` is made of. A round
#: 1.5 MB so the announcement under test quotes a megabyte figure, which is the
#: shape of every real one — the smallest model in the registry is 3.5 MB and
#: the largest 213 MB.
STAND_IN_BYTES = 1_500_000
STAND_IN_SIZE = "1.5 MB"

#: What the stubbed window "returns", and deliberately not 0. A command that did
#: nothing at all also exits 0, so 0 cannot tell "the window opened and the
#: process took its exit code from it" apart from "nothing happened".
WINDOW_EXIT_CODE = 43


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the GUI entry point and record whether it was reached.

    Returns the list the stub appends to, so a test reads ``assert window`` for
    "the window opened" and ``assert not window`` for "it did not".

    ``gui()`` does its ``from alchemypose.gui.app import main`` inside the
    function body, so a stand-in module registered under that name in
    ``sys.modules`` intercepts it and no Qt is imported — which is the whole
    reason the import is written where it is, and is asserted in its own right
    below. Patching ``alchemypose.cli.gui`` instead would be shorter and would
    prove less: it would stop testing that the bare command reaches the real
    entry point, which is the half of this that a refactor can break.
    """
    opened: list[str] = []
    stub = types.ModuleType("alchemypose.gui.app")

    def main() -> int:
        opened.append("opened")
        return WINDOW_EXIT_CODE

    stub.main = main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "alchemypose.gui.app", stub)
    return opened


@pytest.fixture
def cold_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty model cache of this test's own, and a download that stays local.

    Every registry entry is re-pointed at a fixed block of stand-in bytes, so a
    command can be taken through the whole cold-start path — announcement,
    chunks, checksum, cache — in milliseconds, without a network and without
    going anywhere near the real cache the environment override displaces.

    The stand-in ``_download`` reproduces the contract ``ensure`` documents and
    the messaging depends on: the size first, in a call carrying no bytes, then
    the running count. ``tests/unit/test_weights.py`` is what holds the real
    ``_download`` to that; this is what lets the CLI be tested against it.
    """
    from alchemypose.inference import weights

    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    payloads = {key: (key.encode() + b"\n") * (STAND_IN_BYTES // (len(key) + 1)) for key in weights.REGISTRY}
    payloads = {key: payload.ljust(STAND_IN_BYTES, b".") for key, payload in payloads.items()}
    registry = {
        key: replace(spec, sha256=hashlib.sha256(payloads[key]).hexdigest())
        for key, spec in weights.REGISTRY.items()
    }
    key_for_url = {spec.url: key for key, spec in registry.items()}
    monkeypatch.setattr(weights, "REGISTRY", registry)

    def fake_download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> None:
        payload = payloads[key_for_url[url]]
        if progress is not None:
            progress(0, len(payload))
            for done in range(0, len(payload), 1 << 16):
                progress(min(done + (1 << 16), len(payload)), len(payload))
        dest.write_bytes(payload)

    monkeypatch.setattr(weights, "_download", fake_download)
    return tmp_path


@pytest.fixture
def stub_inference(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Stand in for the engine and the image loader; record what the engine found.

    Both are imported inside ``image``'s body, so replacing them on their own
    modules is enough — and is what the existing missing-directory test does.

    Each construction appends the preset it was given *and the model files that
    were on disk at that moment*, which is the only way to assert that the
    fetching happens ahead of the engine rather than merely happening.
    """
    import numpy as np
    from numpy.typing import NDArray

    from alchemypose.core.schema import PoseFrame, PoseResult
    from alchemypose.core.topology import COCO_WHOLEBODY_133
    from alchemypose.inference import rtmw, weights
    from alchemypose.media import image as image_module

    built: list[tuple[str, tuple[str, ...]]] = []

    class _StubEngine:
        model_name = "stub"

        def __init__(self, preset: str = "performance") -> None:
            built.append((preset, tuple(sorted(p.name for p in weights.cache_dir().glob("*.onnx")))))

        def infer(self, image: NDArray[np.uint8], source: str) -> PoseResult:
            frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(8, 8), people=())
            return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="stub")

    monkeypatch.setattr(rtmw, "RTMWEngine", _StubEngine)
    monkeypatch.setattr(
        image_module, "load_image", lambda path: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    return built


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
        # Nothing about a ``--help`` should be able to block, but the child runs
        # the real CLI with a real Qt available, and a callback that opened the
        # window ahead of a subcommand would reach the event loop through
        # ``image --help`` and never come back. That is not hypothetical: it is
        # what this test did when the ``ctx.invoked_subcommand`` guard was
        # removed as an experiment. A hang here is unreadable — the run stops
        # with no failure and no output — so it is bounded.
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert "quantum" in proc.stdout, proc.stdout
    for preset in ("lite", "balanced", "performance"):
        assert preset in proc.stdout


@pytest.mark.parametrize("argv", [[], ["gui"]], ids=["bare", "gui"])
def test_the_window_opens_with_no_command_and_with_gui(argv: list[str], window: list[str]) -> None:
    """``alchemypose`` and ``alchemypose gui`` are the same thing.

    The bare form is what someone types straight after ``pip install
    alchemypose``, and until this existed it printed a page of help — the app
    was installed and the command that names it did not run it.

    Both forms are asserted together, and on the exit code rather than only on
    the fact that something was called, because the two ways this can rot are
    opposite. Losing the bare form is the obvious one: putting
    ``no_args_is_help=True`` back on the ``Typer`` beats
    ``invoke_without_command`` outright — click reads that flag in
    ``Group.parse_args``, before the callback is reachable at all — so the help
    would come back and nothing else here would notice. Losing ``gui`` is the
    quiet one: it is the scriptable name, and dropping the command in favour of
    "just run it bare" would break every script that spells it out.

    The exit code has to be the window's own. ``main()`` returns what
    ``QApplication.exec()`` returned, and a shell that waits on the app is
    entitled to it.
    """
    result = runner.invoke(app, argv)

    assert window, "the GUI entry point was never reached"
    assert result.exit_code == WINDOW_EXIT_CODE, result.output


def test_help_prints_help_and_opens_nothing(window: list[str]) -> None:
    """``--help`` still answers, now that answering nothing means a window.

    click resolves ``--help`` as an eager option during parsing, so it runs
    before the callback can decide anything. That is worth pinning rather than
    trusting: the failure it guards against — reading the help and getting an
    application — is one a user cannot undo or ignore.
    """
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
    for command in ("gui", "image", "models", "version"):
        assert command in result.output
    assert not window


@pytest.mark.parametrize("argv", [["--version"], ["-V"], ["version"]], ids=["flag", "short", "command"])
def test_asking_for_the_version_prints_it_and_opens_nothing(argv: list[str], window: list[str]) -> None:
    """Every spelling of "which version is this?" answers in one line.

    ``--version`` is an eager option on the callback and ``version`` is a
    subcommand; they reach the same string by different routes, and a person
    typing one of them has no reason to know which route their habit takes.
    """
    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert result.output.strip() == __version__
    assert not window


def test_an_unknown_command_is_a_usage_error_and_opens_nothing(window: list[str]) -> None:
    """``alchemypose imagee`` must fail, not launch.

    This is the regression the arrangement most invites. ``invoke_without_command``
    says "run the callback when there is no subcommand", and a plausible reading
    of that is "run it whenever a subcommand cannot be found" — which would turn
    every typo into a window, silently, with the command the user meant never
    running. It does not, because click's ``Group.invoke`` resolves the name and
    fails on an unknown one before it invokes the group callback at all; but that
    ordering is click's to change and nothing in this package would notice.

    Exit code 2 rather than merely non-zero: that is the conventional status for
    a usage error, and it is what a script wrapping this will branch on.
    """
    result = runner.invoke(app, ["notacommand"])

    assert result.exit_code == 2, result.output
    assert "no such command" in result.output.lower()
    assert "notacommand" in result.output
    assert "Usage" in result.output
    assert not window


@pytest.mark.parametrize(
    "argv",
    [["image", "/definitely/not/here.jpg"], ["models", "list"], ["version"]],
    ids=["image", "models-list", "version"],
)
def test_a_subcommand_does_not_also_open_the_window(argv: list[str], window: list[str]) -> None:
    """A group callback runs for *every* subcommand, not only for none.

    That is the trap on the other side of the one above, and it is worse than it
    first looks. Without the ``ctx.invoked_subcommand is None`` guard,
    ``alchemypose image photo.jpg`` does not open a window *and then* do the
    work: ``gui()`` exits with the window's status, so click never reaches the
    subcommand it had already resolved and the photograph is never analysed.
    Measured on that edit: exit code 43 from a stubbed window and not one byte
    of output. Nothing about the invocation looks wrong from the outside, so the
    only way to catch it is to assert the window's absence from commands that
    have no business opening one.
    """
    runner.invoke(app, argv)

    assert not window


def test_the_scripting_commands_load_no_qt() -> None:
    """``image`` and ``models`` must run on a machine that has no Qt at all.

    Every import of ``alchemypose.gui`` is written inside a function body for
    this reason, and the bare-command callback is the newest place that rule
    could be broken — a ``from alchemypose.gui.app import main`` moved up to the
    module's imports would work perfectly, pass every other test here, and make
    the whole CLI depend on PySide6.

    A subprocess, because the assertion is about what is *absent* from
    ``sys.modules``, and this process has almost certainly imported Qt already
    for ``tests/gui``. The commands are chosen to be the ones that do real work
    without needing a model: a missing file and a cache listing both run the
    command body to completion.

    The ``timeout`` is not belt and braces. Nothing in this child stubs the
    window, so a break that reaches ``gui()`` from one of these commands does
    not merely import Qt — it opens an application and blocks in its event loop,
    and the child never exits. Measured on the two ways this file's subject can
    break: hoisting the GUI import to the top of ``alchemypose/cli.py`` fails
    here in under a second, but deleting the ``ctx.invoked_subcommand`` guard
    from the callback made ``image`` open a real window in a child that then had
    to be killed by hand. The timeout is what makes the second of those a
    failure with a message rather than a suite that hangs for ever, and on CI a
    red test rather than a job killed at the runner's limit with nothing to read.
    """
    code = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from alchemypose.cli import app\n"
        "runner = CliRunner()\n"
        "runner.invoke(app, ['image', '/definitely/not/here.jpg'])\n"
        "runner.invoke(app, ['models', 'list'])\n"
        "runner.invoke(app, ['--help'])\n"
        "leaked = [m for m in sys.modules if m.startswith('PySide6')]\n"
        "assert not leaked, leaked\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    assert proc.returncode == 0, proc.stderr


def test_image_names_the_model_and_its_size_before_the_download_it_is_about_to_wait_for(
    cold_cache: Path, stub_inference: list[tuple[str, tuple[str, ...]]], tmp_path: Path
) -> None:
    """The silence this replaces was the whole defect.

    ``alchemypose image photo.jpg`` on a fresh install printed nothing between
    the prompt and the result, while a quarter of a gigabyte came down a
    connection whose speed nobody had promised. From the outside that is
    exactly what a hung command looks like, and the only remedy on offer was
    Ctrl-C — which throws the partial download away.

    Both halves of the sentence are asserted, because either alone leaves the
    user guessing: the *name* without the size gives no idea how long, and the
    size without the name gives no idea how many more of these there are.
    """
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")

    result = runner.invoke(app, ["image", str(photo), "--preset", "lite"])

    assert result.exit_code == 0, result.output
    for key in ("yolox_nano", "rtmw_m_256x192"):
        assert f"Downloading {key} ({STAND_IN_SIZE})" in result.stderr
    assert str(cold_cache) in result.stderr, "the first download should say where it is being kept"


def test_image_fetches_every_model_before_the_engine_asks_for_one(
    cold_cache: Path, stub_inference: list[tuple[str, tuple[str, ...]]], tmp_path: Path
) -> None:
    """The announcing fetch has to be the one that does the work, not a second one.

    ``RTMWEngine`` calls ``weights.ensure`` in its constructor and says nothing
    while it does. If the CLI's own fetch ran after it — or not at all — the
    bytes would still move in silence and the messages would arrive after the
    wait they were meant to explain. What proves the order is what the engine
    finds on disk the moment it is built: both models, already there.
    """
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")

    result = runner.invoke(app, ["image", str(photo), "--preset", "lite"])

    assert result.exit_code == 0, result.output
    assert len(stub_inference) == 1
    preset, cached_when_built = stub_inference[0]
    assert preset == "lite"
    assert len(cached_when_built) == 2, f"the engine was built with {cached_when_built} on disk"


def test_image_says_nothing_when_the_models_are_already_cached(
    cold_cache: Path, stub_inference: list[tuple[str, tuple[str, ...]]], tmp_path: Path
) -> None:
    """The warm run is the common one, and it has to stay exactly as quiet as it was.

    A cache hit costs a checksum and returns; ``ensure`` reports nothing,
    because there is nothing to report. Anything printed here — even a cheerful
    "using cached models" — would be new noise on every single run of a command
    whose output is meant to be piped, in exchange for news the user already
    had.
    """
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")

    first = runner.invoke(app, ["image", str(photo), "--preset", "lite"])
    second = runner.invoke(app, ["image", str(photo), "--preset", "lite"])

    assert first.exit_code == 0 and second.exit_code == 0, first.output + second.output
    assert "Downloading" in first.stderr
    assert second.stderr == "", f"a warm run must print nothing: {second.stderr!r}"


def test_a_redirected_run_gets_plain_sentences_and_no_terminal_control_codes(
    cold_cache: Path, stub_inference: list[tuple[str, tuple[str, ...]]], tmp_path: Path
) -> None:
    """``2> download.log`` has to be readable, which rules out a bar.

    A progress bar is cursor movement, line erasure and colour — none of which
    means anything in a file, and all of which a log viewer renders as rubbish.
    rich's own answer for a non-terminal console is not enough on its own
    either: it drops the escapes but still draws the bar in box-drawing
    characters, and only once the transfer it was tracking is over. So the
    absence of both is asserted, and the sentence that replaces them with it.
    """
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")

    result = runner.invoke(app, ["image", str(photo), "--preset", "lite"])

    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output, "no escape sequences when stderr is not a terminal"
    assert "\r" not in result.output, "nothing may rewrite a line that a file cannot un-write"
    assert "━" not in result.output, "rich's non-terminal bar must not be what gets printed"
    assert f"Downloading yolox_nano ({STAND_IN_SIZE}) ..." in result.output


def test_the_result_goes_to_stdout_and_the_download_news_to_stderr(
    cold_cache: Path, stub_inference: list[tuple[str, tuple[str, ...]]], tmp_path: Path
) -> None:
    """``alchemypose image photo.jpg > out.json`` must still produce JSON and only JSON.

    The whole point of putting the messages on stderr is that the useful
    stream stays machine-readable on the run that needs them most: the first
    one, on a machine with an empty cache.
    """
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")

    result = runner.invoke(app, ["image", str(photo), "--preset", "lite"])

    assert result.exit_code == 0, result.output
    assert "Downloading" not in result.stdout
    assert json.loads(result.stdout)["model"] == "stub"


def test_models_fetch_names_the_size_of_what_it_is_downloading(cold_cache: Path) -> None:
    """The command that already said *which* now also says *how big*.

    ``Fetching <key> ...`` has always named the model, and still does, on
    stdout, whether or not that model turns out to need fetching. What it could
    never say is what the wait was going to cost, because only the response
    headers know — so that half arrives on stderr, and only when bytes move.
    """
    result = runner.invoke(app, ["models", "fetch", "--preset", "lite"])

    assert result.exit_code == 0, result.output
    for key in ("yolox_nano", "rtmw_m_256x192"):
        assert f"Fetching {key} ..." in result.stdout
        assert f"Downloading {key} ({STAND_IN_SIZE})" in result.stderr
    assert result.stdout.strip().endswith("Done.")


def test_models_fetch_keeps_naming_models_it_did_not_have_to_download(cold_cache: Path) -> None:
    """A second fetch is quiet about transfers and still accounts for every model.

    This is the one command whose subject *is* the cache, so "nothing to do for
    this one" is an answer rather than noise — unlike ``image``, where the same
    line would be printed on every run for ever.
    """
    runner.invoke(app, ["models", "fetch", "--preset", "lite"])

    result = runner.invoke(app, ["models", "fetch", "--preset", "lite"])

    assert result.exit_code == 0, result.output
    for key in ("yolox_nano", "rtmw_m_256x192"):
        assert f"Fetching {key} ..." in result.stdout
    assert result.stderr == "", f"nothing was downloaded, so nothing may be said about downloading: {result.stderr!r}"


def test_image_reports_a_failed_download_as_a_sentence(
    cold_cache: Path, stub_inference: list[tuple[str, tuple[str, ...]]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fetching from the command moved a failure earlier; it must still land softly.

    ``RTMWEngine`` used to be where the download happened and where a
    ``WeightsError`` came from, and the command already turned that into one
    line. Now the fetch happens a few lines above it, outside the constructor —
    which is only safe because it is inside the same ``try``. A network that
    drops halfway through 213 MB is the single most likely way this command
    ever fails, and a traceback is not an answer to it.
    """
    from alchemypose.errors import WeightsError
    from alchemypose.inference import weights

    def refuse(url: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> None:
        if progress is not None:
            progress(0, 213_433_855)  # the size is announced, then the connection dies
        raise WeightsError(f"could not download {url}: [Errno 54] Connection reset by peer")

    monkeypatch.setattr(weights, "_download", refuse)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"")

    result = runner.invoke(app, ["image", str(photo), "--preset", "lite"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Error: could not download" in result.output
    assert "Downloading yolox_nano (213.4 MB)" in result.stderr
    assert not stub_inference, "the engine must not be built once the weights are known to be missing"


# ------------------------------------------------------------------------ video


@pytest.fixture
def _video_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    """A fake engine and reader, a private cache, and weights that are already 'fetched'."""
    from alchemypose import cli as cli_module
    from alchemypose.inference import rtmw
    from alchemypose.media import video as video_module
    from tests.support.fakes import FakeFrameEngine, FakeReader

    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli_module, "_fetch_weights", lambda preset: None)
    built: list[str] = []

    def engine(preset: str = "performance") -> FakeFrameEngine:
        built.append(preset)
        return FakeFrameEngine()

    monkeypatch.setattr(rtmw, "RTMWEngine", engine)
    monkeypatch.setattr(video_module, "open_video", lambda path, backend=None: FakeReader(frames=6, path=str(path)))
    clip = tmp_path / "dance.mp4"
    clip.write_bytes(b"stand-in")
    return clip, built


def test_video_writes_every_frame_with_track_ids(_video_stubs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.core.serialize import from_json

    clip, built = _video_stubs
    out = tmp_path / "out.json"
    result = runner.invoke(app, ["video", str(clip), "--preset", "lite", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert built == ["lite"]
    parsed = from_json(out.read_text())
    assert [f.frame_index for f in parsed.frames] == list(range(6))
    assert {p.track_id for f in parsed.frames for p in f.people} == {0}
    assert "6 frames, 1 tracked person" in result.stderr


def test_video_a_second_run_comes_from_the_cache(_video_stubs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    clip, built = _video_stubs
    runner.invoke(app, ["video", str(clip), "--preset", "lite", "--json", str(tmp_path / "a.json")])
    second = runner.invoke(app, ["video", str(clip), "--preset", "lite", "--json", str(tmp_path / "b.json")])
    assert second.exit_code == 0
    assert built == ["lite"], "no engine is built for a cached clip"
    assert "from cache" in second.stderr


def test_video_reports_a_missing_file_without_a_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["video", str(tmp_path / "missing.mp4")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_gltf_on_a_preset_without_depth_is_refused_before_any_inference(_video_stubs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    clip, built = _video_stubs
    result = runner.invoke(app, ["video", str(clip), "--preset", "lite", "--gltf", str(tmp_path / "a.glb")])
    assert result.exit_code == 1
    assert "--preset 3d" in result.output
    assert built == []
