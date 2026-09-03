"""Guards the environment the suite itself needs, which no other test observes.

``ORT_DISABLE_TELEMETRY=1`` is the one setting whose absence pytest cannot
report. onnxruntime's POSIX telemetry uploader runs on its own thread, and at
interpreter exit the teardown of the telemetry log manager can free a
``recursive_mutex`` that an in-flight upload then tries to lock. The lock fails
with ``EINVAL``, libc++ raises, and the process dies of SIGABRT::

    libc++abi: terminating due to uncaught exception of type
    std::__1::system_error: recursive_mutex lock failed: Invalid argument

The crash lands *after* the run has finished and printed its pass line, so
pytest's own result says "passed" either way: the only visible difference is a
nonzero exit status on the runs that happen to lose the race. It is a race
against the network and it does not fire every time — measured over 40 runs of
the whole suite per arm, 0 aborts with the variable set and 5 without. That
combination, silent and intermittent, is why deleting the line in
``tests/conftest.py`` would otherwise go unnoticed for as long as the dice
allowed.

The value is read once, when onnxruntime's telemetry stack initialises, so it
counts only if it is already set at the first ``import onnxruntime`` in the
process. This module therefore reads it at *import* time — during collection,
before any test body runs — so a setting that only arrived later, from a
fixture say, cannot make the assertion pass.

Nothing else in the environment is asserted here on purpose. ``QT_QPA_PLATFORM``
looks like a peer but is not one: it is set in ``tests/gui/conftest.py``, which
pytest loads only when it collects ``tests/gui``, so an assertion here would
fail merely because someone ran ``pytest tests/unit``. Its breakage is loud in
any case — a missing offscreen platform aborts the GUI tests with a plugin error
or opens real windows, both of which announce themselves. Only invariants that
are load-bearing *and* silent belong in this file.
"""

import os

#: Read at import (collection) time, not inside the test — see the module docstring.
_ORT_DISABLE_TELEMETRY = os.environ.get("ORT_DISABLE_TELEMETRY")


def test_onnxruntime_telemetry_is_disabled_for_the_whole_run() -> None:
    assert _ORT_DISABLE_TELEMETRY == "1", (
        f"ORT_DISABLE_TELEMETRY is {_ORT_DISABLE_TELEMETRY!r}, expected '1'. "
        "tests/conftest.py must set it before anything imports onnxruntime, or the interpreter "
        "can abort with 'recursive_mutex lock failed' after the run has already printed its pass line."
    )
