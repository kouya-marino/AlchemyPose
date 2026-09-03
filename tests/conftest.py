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
"""

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
