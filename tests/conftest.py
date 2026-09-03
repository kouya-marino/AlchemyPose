"""Suite-wide setup that has to happen before anything imports onnxruntime.

onnxruntime's POSIX build carries a telemetry client that batches events and
uploads them over HTTP on a thread of its own. Its shutdown is not safe against
that uploader. At ``exit()`` the static destructor of ``PosixEnv`` tears the
telemetry log manager down on the main thread, and if an upload round-trip is
still in flight the uploader's decoder goes on to lock a ``recursive_mutex``
whose storage that teardown has already freed. The lock fails with ``EINVAL``,
libc++ turns it into an uncaught ``std::system_error``, and the process dies of
SIGABRT — *after* the run has finished and printed its pass line::

    libc++abi: terminating due to uncaught exception of type
    std::__1::system_error: recursive_mutex lock failed: Invalid argument

So it is a race against the network, and the window is narrow: the exit has to
land while a request is outstanding. A process that exits promptly never sends
one, and a long-lived one has usually seen the response come back. A run of the
whole suite lands in that window often enough to abort roughly once in ten,
which is why no single test directory ever showed it alone — each finishes
before the first upload leaves.

``ORT_DISABLE_TELEMETRY`` is read once, when the telemetry stack initialises,
so it only counts if it is set before the first ``import onnxruntime`` anywhere
in the process. The root conftest is imported ahead of every test module, which
is early enough. ``onnxruntime.disable_telemetry_events()`` is not a substitute:
it stops later events being logged but leaves the log manager and its uploader
already running, and a process calling it still aborts at the same rate.

``setdefault`` leaves an explicit setting alone, so a run that wants telemetry
back can just ask for it in the environment.

The ordering is the fragile half, and it breaks quietly in both directions.
Setting the variable after the first import is a no-op rather than an error:
onnxruntime has already read it, the uploader is already running, and nothing
raises or warns. The edit that gets it wrong is an ordinary one — a helper
added to this file that imports, say, ``alchemypose.inference.session``, put up
with the other imports at the top where imports belong, drags onnxruntime in
above the ``setdefault`` below. Linting does not catch that either: ruff's E402
exempts an import placed after an ``os.environ`` mutation, so both orderings
pass ``ruff check`` and only the broken one is silent. Measured on this
machine, by counting the process's OS threads after importing onnxruntime: set
before the import, 1; set after it, 3 — the same as never setting it at all.

``ORT_ALREADY_IMPORTED`` is what makes that observable. Once the interpreter
has moved on, no later code can reconstruct whether the variable won the race,
because the environment looks identical either way. Sampling it on the line
above the ``setdefault`` is the only moment the answer still exists, and
``tests/unit/test_suite_environment.py`` asserts on it.
"""

import os
import sys

#: Whether onnxruntime had already been imported by the time this module got to
#: set the variable below — the fact the environment stops recording a moment
#: later. Asserted in ``tests/unit/test_suite_environment.py``.
ORT_ALREADY_IMPORTED = "onnxruntime" in sys.modules

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
