"""Robots: the Unitree G1's model, its kinematics, and putting human motion on it.

Everything under here is Qt-free and needs no OpenGL, like ``core`` and
``inference``: the robot is a set of numpy arrays until something draws it.
Nothing here talks to a real robot. It is kinematic visualisation and
reference data, not control.
"""
