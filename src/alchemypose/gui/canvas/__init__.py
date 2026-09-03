"""The photograph, and the skeleton drawn over it.

``view`` is the zoomable, pannable image; ``scene`` turns a result into the
items above it; ``items`` holds the two drawing primitives; ``interaction``
maps clicks onto emphasis and solo; ``lod`` decides how much face a given zoom
earns. Everything here is registered to the photograph in image-pixel scene
coordinates, and none of it needs a window to run.
"""
