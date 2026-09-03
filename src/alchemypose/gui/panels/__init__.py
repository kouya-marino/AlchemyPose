"""The side panels: what is drawn, whose body it is, and what it means.

``layers`` owns the visibility tree and is the single source of truth for what
the canvas draws; ``presets`` sets that tree from one click and ``people``
chooses whose parts it applies to. ``inspector`` reads back the other way — the
keypoint under the cursor, and the colour legend.
"""
