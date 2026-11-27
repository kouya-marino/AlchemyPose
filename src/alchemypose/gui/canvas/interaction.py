"""Select versus solo.

Clicking a joint emphasises its group and fades the rest — reversible, nothing
lost. Soloing (double-click, or ``Shift+H``) hides the rest, which is
destructive enough that it needs its own gesture and a remembered previous
state to return to. ``Esc`` / ``Alt+H`` undo a solo and drop the emphasis.

The emphasis fade goes through ``PoseOverlay.set_emphasis`` rather than being
painted onto the items from here. The overlay rebuilds every item on any
selection or threshold change, so a dim written directly onto items would be
wiped by the very next layer toggle; letting the overlay *record* the emphasis
means a rebuild reproduces it. One mechanism, not two that disagree.

Visibility is the other half, and it is deliberately *not* symmetrical with
emphasis: soloing never touches the overlay's selection. It writes to
``LayerPanel.set_selection`` and stops there. The tree emits
``selection_changed``, and whoever owns the window turns that into the
overlay's selection — which is the only place that knows the rest of the
picture, most importantly which person the chips have narrowed the view to.
A push from here would have to guess at that narrowing, and guessing "no
narrowing" silently puts everybody back on screen while the chip still claims
one person. So: emphasis is ours to push, visibility is the tree's to announce.
"""

from __future__ import annotations

from alchemypose.core.schema import KeypointSelection
from alchemypose.core.topology import RENDER_GROUPS
from alchemypose.gui.canvas.scene import PoseOverlay
from alchemypose.gui.panels.layers import LayerPanel


class SelectionController:
    """Owns 'what is emphasised' and 'what is soloed' for one canvas."""

    def __init__(self, overlay: PoseOverlay, layers: LayerPanel) -> None:
        self._overlay = overlay
        self._layers = layers
        self._emphasised: str | None = None
        self._soloed: str | None = None
        self._before_solo: KeypointSelection | None = None

    @property
    def emphasised(self) -> str | None:
        return self._emphasised

    @property
    def soloed(self) -> str | None:
        return self._soloed

    def select_group(self, group: str) -> None:
        """Fade every other group. Nothing is hidden, nothing is remembered."""
        self._emphasised = group
        self._overlay.set_emphasis(group)
        self._overlay.set_image_dimmed(True)

    def solo_group(self, group: str) -> None:
        """Hide everything else, remembering what to come back to.

        ``_before_solo`` is only captured the first time — soloing a second
        group while already soloed must not overwrite the original selection
        with the (already-narrowed) soloed one, or ``clear()`` would restore
        the wrong thing.
        """
        if self._before_solo is None:
            self._before_solo = self._layers.current_selection()
        self._soloed = group
        self._emphasised = group
        self._layers.set_selection(KeypointSelection(visible=frozenset(RENDER_GROUPS[group])))
        self._overlay.set_emphasis(None)
        self._overlay.set_image_dimmed(True)

    def clear(self) -> None:
        """Esc / Alt+H: drop the emphasis and undo any solo."""
        if self._before_solo is not None:
            self._layers.set_selection(self._before_solo)
            self._before_solo = None
        self._soloed = None
        self._emphasised = None
        self._overlay.set_emphasis(None)
        self._overlay.set_image_dimmed(False)
