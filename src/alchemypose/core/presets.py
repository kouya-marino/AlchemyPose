"""Named visibility presets.

Presets are not a mode. Each one is a set of render-group names; the UI applies
one by setting the layer tree's check states, so the tree always shows the truth
about what is drawn. ``matching_preset`` is the inverse, letting the UI light up
a chip when hand-toggling happens to land on it.
"""

from __future__ import annotations

from collections.abc import Iterable

from alchemypose.core.schema import KeypointSelection
from alchemypose.core.topology import RENDER_GROUPS

_ALL_GROUPS = tuple(RENDER_GROUPS)

#: preset name -> the render groups it turns on
PRESETS: dict[str, tuple[str, ...]] = {
    "full_body": ("torso_head", "left_arm", "right_arm", "left_leg", "right_leg", "left_foot", "right_foot"),
    "upper": ("torso_head", "left_arm", "right_arm", "left_hand", "right_hand"),
    "lower": ("left_leg", "right_leg", "left_foot", "right_foot"),
    "left_side": ("left_arm", "left_leg", "left_foot", "left_hand"),
    "right_side": ("right_arm", "right_leg", "right_foot", "right_hand"),
    "hands": ("left_hand", "right_hand"),
    "face": ("face_jaw", "face_brows", "face_nose", "face_eyes", "face_lips"),
    "all": _ALL_GROUPS,
    "none": (),
}


def selection_from_groups(groups: Iterable[str]) -> KeypointSelection:
    """Union the indices of the named render groups."""
    indices: set[int] = set()
    for name in groups:
        indices.update(RENDER_GROUPS[name])
    return KeypointSelection(visible=frozenset(indices))


def selection_for(preset: str) -> KeypointSelection:
    """The selection a preset chip applies. Raises KeyError for an unknown name."""
    return selection_from_groups(PRESETS[preset])


# matching_preset (below) returns the first exact match in dict order, so two
# presets resolving to the same visible set would make the second one
# unreachable and the UI would silently light up the wrong chip. Nothing in
# PRESETS collides today, but a future preset added without checking this
# could collide silently at runtime instead of loudly at import time — so
# assert distinctness once, here, over the (cheap: nine small sets) set of
# all presets.
_visible_sets = [selection_for(name).visible for name in PRESETS]
assert len(_visible_sets) == len(set(_visible_sets)), (
    "PRESETS must all resolve to distinct visible sets: matching_preset picks the first "
    "exact match in dict order, so a collision would silently hide one preset behind another"
)


def matching_preset(selection: KeypointSelection) -> str | None:
    """The preset this selection exactly equals, if any.

    Order matters only for presets that could coincide; ``none`` and ``all`` are
    checked like any other because an empty or complete selection is unambiguous.
    """
    for name in PRESETS:
        if selection_for(name).visible == selection.visible:
            return name
    return None
