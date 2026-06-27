"""
调试IK控制器 - 检查每一步的状态变化。

Usage::
    python debug_ik_controller.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env.franka_env import FrankaEnv
from controllers.ik_controller import IKController


def main():
    env = FrankaEnv(render_mode='human')
    env.reset()

    controller = IKController(env)

    print("初始状态:")
    print(f"  关节位置: {env.arm_joint_positions}")
    print(f"  末端位置: {env.endeffector_position}")
    print(f"  方块位置: {env.block_position}")
    print(f"  目标位置: {env.target_position}")
    print(f"  距离目标: {env.distance_to_target}")

    # 运行10步
    for step in range(10):
        q_before = env.arm_joint_positions.copy()
        ee_before = env.endeffector_position.copy()
        ctrl_before = [env.data.ctrl[aid] for aid in env._arm_actuator_ids]

        controller.compute_control()

        ctrl_after = [env.data.ctrl[aid] for aid in env._arm_actuator_ids]

        env.step()

        q_after = env.arm_joint_positions.copy()
        ee_after = env.endeffector_position.copy()

        print(f"\n步骤 {step}:")
        print(f"  ctrl变化: {[a - b for a, b in zip(ctrl_after, ctrl_before)]}")
        print(f"  q_before: {q_before}")
        print(f"  q_after: {q_after}")
        print(f"  q变化: {q_after - q_before}")
        print(f"  ee_before: {ee_before}")
        print(f"  ee_after: {ee_after}")
        print(f"  ee变化: {ee_after - ee_before}")
        print(f"  距离目标: {env.distance_to_target}")


if __name__ == "__main__":
    main()
