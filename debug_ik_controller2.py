"""
调试IK控制器 - 检查ctrl设置。
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
    print(f"  ctrl: {[env.data.ctrl[aid] for aid in env._arm_actuator_ids]}")
    print(f"  关节位置: {env.arm_joint_positions}")

    # 运行5步
    for step in range(5):
        ctrl_before = [env.data.ctrl[aid] for aid in env._arm_actuator_ids]
        q_before = env.arm_joint_positions.copy()

        controller.compute_control()

        ctrl_after = [env.data.ctrl[aid] for aid in env._arm_actuator_ids]
        q_target = np.array(ctrl_after)  # ctrl就是目标位置

        print(f"\n步骤 {step}:")
        print(f"  ctrl_before: {ctrl_before}")
        print(f"  ctrl_after: {ctrl_after}")
        print(f"  ctrl变化: {[a - b for a, b in zip(ctrl_after, ctrl_before)]}")
        print(f"  q_target - q_before: {q_target - q_before}")

        env.step()

        q_after = env.arm_joint_positions.copy()
        print(f"  q_after: {q_after}")
        print(f"  q_target - q_after: {q_target - q_after}")


if __name__ == "__main__":
    main()
