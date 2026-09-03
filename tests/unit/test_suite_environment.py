"""Guards the environment the suite itself needs, which no other test observes.

An assertion about an environment variable usually is cargo cult: it restates a
line of setup that any real test would notice the absence of. This one is not,
because the thing it protects against cannot reach a test at all.

``ORT_DISABLE_TELEMETRY=1`` is the setting whose absence pytest cannot report.
onnxruntime's POSIX telemetry uploader runs on its own thread, and at
interpreter exit the teardown of the telemetry log manager can free a
``recursive_mutex`` that an in-flight upload then tries to lock. The lock fails
with ``EINVAL``, libc++ raises, and the process dies of SIGABRT::

    libc++abi: terminating due to uncaught exception of type
    std::__1::system_error: recursive_mutex lock failed: Invalid argument

That happens *after* the last test has run and after pytest has printed its
pass line, so there is no point in the run at which an assertion could see it:
the report says "442 passed" on the aborted runs too, and the only trace is a
nonzero exit status. It is a race against the network and it does not fire
every time — measured over 40 runs of the whole suite per arm, 0 aborts with
the variable set and 5 without. Silent and intermittent together are why
deleting the line in ``tests/conftest.py`` would otherwise go unnoticed for as
long as the dice allowed, and why the guard is worth its two assertions.

They are two because there are two ways to lose the setting, and only one of
them is about the value.

The value is the obvious one. onnxruntime parses the variable as a boolean, and
what counts as "disable" is narrower than it looks: ``1``, ``true`` and ``yes``
switch the telemetry stack off, while ``0`` and ``false`` are ordinary,
well-formed ways of asking to keep it *on*. So ``0`` is not a softer ``1`` and
it is not a typo the library will second-guess — it reads exactly like leaving
the variable unset, uploader threads and all.

The order is the one that gets missed. onnxruntime reads the variable once,
when its telemetry stack initialises at the first ``import onnxruntime`` in the
process, and never looks again. Setting it after that import is a silent no-op
rather than an error — the assignment succeeds, ``os.environ`` reports the
value you wanted, and telemetry stays live for the rest of the run. A test that
only checks the value therefore passes on a process whose uploader is running,
which is the exact state the guard exists to rule out. The failing edit is a
plausible one: a helper added to ``tests/conftest.py`` that imports something
under ``alchemypose.inference``, placed at the top with the other imports, is
enough, and it passes ``ruff check`` because E402 exempts an import that
follows an ``os.environ`` mutation. Hence ``ORT_ALREADY_IMPORTED``, sampled in
``tests/conftest.py`` on the line above the ``setdefault``: it is asserted here
because that is the last moment the ordering leaves a trace.

The value is read at *import* time — during collection, before any test body
runs — so a setting that only arrived later, from a fixture say, cannot make
the assertion pass.

Nothing else in the environment is asserted here on purpose. ``QT_QPA_PLATFORM``
looks like a peer but is not one: it is set in ``tests/gui/conftest.py``, which
pytest loads only when it collects ``tests/gui``, so an assertion here would
fail merely because someone ran ``pytest tests/unit``. Its breakage is loud in
any case — a missing offscreen platform aborts the GUI tests with a plugin error
or opens real windows, both of which announce themselves. Only invariants that
are load-bearing *and* silent belong in this file.
"""

import os

from tests.conftest import ORT_ALREADY_IMPORTED

#: Read at import (collection) time, not inside the test — see the module docstring.
_ORT_DISABLE_TELEMETRY = os.environ.get("ORT_DISABLE_TELEMETRY")


def test_onnxruntime_telemetry_is_disabled_for_the_whole_run() -> None:
    assert _ORT_DISABLE_TELEMETRY == "1", (
        f"ORT_DISABLE_TELEMETRY is {_ORT_DISABLE_TELEMETRY!r}, expected '1'. "
        "tests/conftest.py must set it before anything imports onnxruntime, or the interpreter "
        "can abort with 'recursive_mutex lock failed' after the run has already printed its pass line."
    )
    assert ORT_ALREADY_IMPORTED is False, (
        "onnxruntime was already imported when tests/conftest.py set ORT_DISABLE_TELEMETRY "
        f"(the value is {_ORT_DISABLE_TELEMETRY!r}, so the value check above passed and saw nothing wrong). "
        "Setting it late is a silent no-op: telemetry is live for this whole run and the interpreter "
        "can abort with 'recursive_mutex lock failed' after the pass line is printed. Something imported "
        "above the setdefault in tests/conftest.py — move it below, or import it inside the function."
    )
