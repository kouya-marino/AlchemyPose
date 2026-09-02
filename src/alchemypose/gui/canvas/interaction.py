"""Select versus solo.

Clicking a joint emphasises its group and fades the rest — reversible, nothing
lost. Soloing (double-click, or ``Shift+H``) hides the rest, which is
destructive enough that it needs its own gesture and a remembered previous
state to return to. ``Esc`` / ``Alt+H`` undo a solo and drop the emphasis.

The emphasis fade is applied here rather than through a method on
``PoseOverlay`` — it only needs the overlay's already-public ``keypoint_items``
/ ``bone_items`` lists and ``KeypointItem.group_name``, both of which are
already part of its public surface.

Soloing never writes to the overlay directly: it goes through
``LayerPanel.set_selection`` so the tree stays the single source of truth for
what is visible, exactly as every other visibility change in this app does.
"""

from __future__ import annotations

from alchemypose.core.schema import KeypointSelection
from alchemypose.core.topology import RENDER_GROUPS
from alchemypose.gui.canvas.scene import PoseOverlay
from alchemypose.gui.panels.layers import LayerPanel

DIM_OPACITY = 0.25


def _apply_emphasis(overlay: PoseOverlay, group: str | None) -> None:
    """Fade every group except ``group``. Purely visual — nothing is removed."""
    for item in overlay.keypoint_items:
        item.setOpacity(1.0 if group is None or item.group_name == group else DIM_OPACITY)
    for bone in overlay.bone_items:
        bone.setOpacity(DIM_OPACITY if group is not None else 1.0)


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
        _apply_emphasis(self._overlay, group)

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
        self._sync_overlay_from_layers()
        _apply_emphasis(self._overlay, None)

    def clear(self) -> None:
        """Esc / Alt+H: drop the emphasis and undo any solo."""
        if self._before_solo is not None:
            self._layers.set_selection(self._before_solo)
            self._before_solo = None
            self._sync_overlay_from_layers()
        self._soloed = None
        self._emphasised = None
        _apply_emphasis(self._overlay, None)

    def _sync_overlay_from_layers(self) -> None:
        """Push the tree's own resulting state to the overlay.

        Always read back through ``LayerPanel.current_selection()`` rather than
        forwarding the value that was just written — the tree stays the single
        source of truth for what is visible, and the overlay only ever mirrors
        it, never diverges from it.
        """
        self._overlay.set_selection(self._layers.current_selection())
