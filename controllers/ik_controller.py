from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import mujoco

from env.franka_env import FrankaEnv
from .base_controller import BaseController


class Stage(Enum):
    """Pick-and-place task stages."""
    APPROACH = auto()      # Move above block with open gripper
    DESCEND = auto()       # Lower to grasp position
    GRASP = auto()         # Close gripper
    LIFT = auto()          # Lift the block
    MOVE = auto()          # Move to target area
    PLACE = auto()         # Place the block
    RETREAT = auto()       # Move away
    DONE = auto()          # Task complete


class IKController(BaseController):
    """Inverse kinematics controller for pick-and-place task.

    Uses online IK with mj_kinematics (no physics stepping) for Jacobian
    computation, continuously tracking target positions. This avoids
    pushing the block during IK iteration.
    """

    def __init__(self, env: FrankaEnv) -> None:
        super().__init__()
        self.env: FrankaEnv = env
        self.model = env.model
        self.data = env.data

        # Task state
        self.stage: Stage = Stage.APPROACH

        # Target positions (tracked online via IK)
        self.target_pos = np.array([0.45, 0.16, 0.40])  # Start above block

        # Stage timing (simulation steps)
        self.stage_steps = {
            Stage.APPROACH: 800,
            Stage.DESCEND: 800,
            Stage.GRASP: 600,
            Stage.LIFT: 800,
            Stage.MOVE: 1500,
            Stage.PLACE: 600,
            Stage.RETREAT: 500,
        }

        # State
        self.stage_counter: int = 0

        # IK parameters
        self.ik_gain = 0.8
        self.ik_damping = 0.005
        self.ik_max_dq = 0.02

    def reset(self) -> None:
        """Reset controller state."""
        super().reset()
        self.stage = Stage.APPROACH
        self.stage_counter = 0
        self.target_pos = np.array([0.45, 0.16, 0.40])

    def is_done(self) -> bool:
        """Check if task is complete."""
        return self.stage == Stage.DONE

    def _solve_ik_step(self, target_pos: np.ndarray) -> None:
        """One IK step using mj_kinematics (no physics).

        Sets qpos directly, runs mj_kinematics, computes Jacobian,
        and updates the target joint angles for the PD controller.
        """
        # Read current joint positions from PD target (ctrl)
        q = self.env.arm_joint_positions.copy()

        # Set qpos directly for FK (doesn't affect physics)
        for i, adr in enumerate(self.env._arm_qpos_adrs):
            self.data.qpos[adr] = q[i]

        # Run FK only
        mujoco.mj_kinematics(self.model, self.data)

        # Current hand position
        ee = self.data.xpos[self.env.hand_body_id].copy()
        error = target_pos - ee

        if np.linalg.norm(error) < 1e-4:
            return  # Already at target

        # Compute Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.env.hand_body_id)
        J = jacp[:, self.env.arm_dof_adrs]

        # DLS solution
        JJT = J @ J.T + self.ik_damping ** 2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error) * self.ik_gain

        # Clip for stability
        dq = np.clip(dq, -self.ik_max_dq, self.ik_max_dq)

        # Update target
        q_new = q + dq

        # Joint limits
        for i in range(4):
            jnt_id = self.env._arm_joint_ids[i]
            lo = self.model.jnt_range[jnt_id, 0]
            hi = self.model.jnt_range[jnt_id, 1]
            q_new[i] = np.clip(q_new[i], lo, hi)

        self.env.set_arm_target(q_new)

    def compute_control(self) -> None:
        """Compute one control step."""
        if self.stage == Stage.DONE:
            return

        self.stage_counter += 1

        # Update target position for current stage
        self._update_target()

        # Solve IK step (using mj_kinematics, no physics)
        self._solve_ik_step(self.target_pos)

        # Manage gripper
        self._update_gripper()

        # Stage transition
        if self.stage_counter >= self.stage_steps.get(self.stage, 300):
            self._transition_stage()

        self._step_counter += 1

    def _update_target(self) -> None:
        """Update target position based on current stage."""
        block_pos = self.env.block_position

        if self.stage == Stage.APPROACH:
            # Track actual block position for approach
            self.target_pos = np.array([block_pos[0], block_pos[1], 0.40])
        elif self.stage == Stage.DESCEND:
            # Use actual block position
            self.target_pos = np.array([block_pos[0], block_pos[1], 0.278])
        elif self.stage == Stage.GRASP:
            # Keep at grasp position while closing gripper
            self.target_pos = np.array([block_pos[0], block_pos[1], 0.278])
        elif self.stage == Stage.LIFT:
            self.target_pos = np.array([block_pos[0], block_pos[1], 0.45])
        elif self.stage == Stage.MOVE:
            self.target_pos = np.array([0.45, -0.16, 0.45])
        elif self.stage == Stage.PLACE:
            self.target_pos = np.array([0.45, -0.16, 0.378])
        elif self.stage == Stage.RETREAT:
            self.target_pos = np.array([0.45, -0.16, 0.50])

    def _update_gripper(self) -> None:
        """Update gripper state based on stage."""
        if self.stage in (Stage.APPROACH, Stage.DESCEND):
            self.env.set_gripper(1.0)  # Open
        elif self.stage == Stage.GRASP:
            self.env.set_gripper(0.0)  # Close
        elif self.stage in (Stage.LIFT, Stage.MOVE):
            self.env.set_gripper(0.0)  # Keep closed
        elif self.stage in (Stage.PLACE, Stage.RETREAT):
            self.env.set_gripper(1.0)  # Open

    def _transition_stage(self) -> None:
        """Transition to next stage."""
        order = [
            Stage.APPROACH, Stage.DESCEND, Stage.GRASP, Stage.LIFT,
            Stage.MOVE, Stage.PLACE, Stage.RETREAT, Stage.DONE
        ]
        idx = order.index(self.stage)
        if idx + 1 < len(order):
            self.stage = order[idx + 1]
            self.stage_counter = 0
            print(f"  [{self.step_count:4d}] → {self.stage.name}")
