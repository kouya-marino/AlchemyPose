# src/alchemypose/errors.py
"""The exceptions AlchemyPose raises on its own behalf."""


class AlchemyPoseError(Exception):
    """Base class, so callers can catch everything from this package."""


class WeightsError(AlchemyPoseError):
    """A model file could not be fetched, or failed verification."""


class InferenceError(AlchemyPoseError):
    """A model ran but could not produce a usable result."""
