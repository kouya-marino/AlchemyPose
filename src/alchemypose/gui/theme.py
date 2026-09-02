"""Every colour in one file.

The side convention is load-bearing: green is the *person's* left, which is the
viewer's right in a face-on photo. The legend must say so, because getting this
backwards is the single most common misreading of a pose overlay.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

CANVAS_BG = "#0e0f13"
PANEL_BG = "#16181d"
TEXT = "#d7dae0"
TEXT_DIM = "#8b909a"

LEFT = "#3ddc84"      # person's left
RIGHT = "#ff9f43"     # person's right
MIDLINE = "#4dd0e1"   # torso, head, spine
FACE = "#c9cdd6"
ACCENT = "#7c6cff"

#: Opacity applied to the photograph once anything is selected, so the
#: skeleton reads as the subject rather than competing with the image.
DIMMED_IMAGE_OPACITY = 0.6

GROUP_COLORS: dict[str, str] = {
    "torso_head": MIDLINE,
    "left_arm": LEFT,
    "right_arm": RIGHT,
    "left_leg": LEFT,
    "right_leg": RIGHT,
    "left_foot": LEFT,
    "right_foot": RIGHT,
    "left_hand": LEFT,
    "right_hand": RIGHT,
    "face_jaw": FACE,
    "face_brows": FACE,
    "face_nose": FACE,
    "face_eyes": FACE,
    "face_lips": FACE,
}

SIDE_LEGEND: tuple[tuple[str, str], ...] = (
    ("Person's left", LEFT),
    ("Person's right", RIGHT),
    ("Midline", MIDLINE),
)


def color_for(group: str) -> QColor:
    """The colour for a render group; a neutral grey for anything unrecognised."""
    return QColor(GROUP_COLORS.get(group, TEXT_DIM))


STYLESHEET = f"""
QMainWindow, QWidget {{ background: {PANEL_BG}; color: {TEXT}; }}
QStatusBar {{ color: {TEXT_DIM}; }}
QTreeWidget {{ background: {PANEL_BG}; border: none; outline: none; }}
QTreeWidget::item {{ padding: 3px 0; }}
QToolButton {{
    background: transparent; color: {TEXT_DIM};
    border: 1px solid #2a2e37; border-radius: 11px; padding: 3px 10px;
}}
QToolButton:checked {{ color: {TEXT}; border-color: {ACCENT}; background: #22202e; }}
"""
