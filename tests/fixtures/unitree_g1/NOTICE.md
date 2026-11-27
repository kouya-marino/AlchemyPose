# Unitree G1 fixture

- **`g1_29dof_rev_1_0.urdf`** — copied unmodified from
  [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros),
  `robots/g1_description/`, at commit `24d6307f85986fd489807fb54e0995a798c36b37`.
  BSD-3-Clause, © Unitree Robotics; the licence is in `LICENSE` beside it, as its
  terms require. The meshes it references are *not* here: the application fetches
  them at runtime (`alchemypose.robot.assets`), and the tests that use this file
  need only the kinematic tree.
- **`fk_golden.json`** — every link's world pose for three joint configurations
  (all zeros and two random in-limit ones), computed by MuJoCo loading that same
  URDF. Independent ground truth for `alchemypose.robot.urdf`.
