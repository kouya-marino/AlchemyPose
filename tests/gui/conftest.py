import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.gui


@pytest.fixture
def reporter():  # type: ignore[no-untyped-def]
    """A Reporter that records instead of opening modals.

    Constructing a real modal while a worker thread is alive has segfaulted in
    sibling projects; tests must never do it.
    """
    from alchemypose.gui.reporting import SilentReporter

    return SilentReporter()
