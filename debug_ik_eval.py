"""
调试IK学习评估 - 检查每一步的状态变化。

Usage::
    python debug_ik_eval.py --model_path data/20260626_005525/checkpoints_ik/model_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env.franka_env import FrankaEnv


class IKModel(torch.nn.Module):
    def __init__(self, input_dim=9, output_dim=4):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
            torch.nn.Tanh(),
        )
        self.output_scale = 0.02

    def forward(self, x):
        return self.model(x) * self.output_scale


def build_features(env):
    q_current = env.arm_joint_positions
    ee_pos = env.endeffector_position
    blk_pos = env.block_position
    tgt_pos = env.target_position
    finger_open = float(np.mean(env.finger_joint_positions))
    dist_ee_block = float(np.linalg.norm(ee_pos - blk_pos))

    if dist_ee_block < 0.15 and finger_open < 0.02:
        target_pos = tgt_pos
    else:
        target_pos = blk_pos

    pos_error = target_pos - ee_pos
    dist_to_target = np.linalg.norm(pos_error)

    if dist_ee_block > 0.3:
        phase = 0.0
    elif dist_ee_block > 0.15:
        phase = 0.2
    elif finger_open < 0.02:
        phase = 0.5
    else:
        phase = 0.8

    features = np.concatenate([
        q_current,
        pos_error,
        [dist_to_target],
        [phase],
    ]).astype(np.float32)

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    model = IKModel()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=True))
    model.eval()

    env = FrankaEnv(render_mode='human')
    env.reset()

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

        features = build_features(env)
        features_tensor = torch.from_numpy(features).unsqueeze(0)

        with torch.no_grad():
            dq = model(features_tensor).squeeze(0).numpy()

        q_target = q_before + dq
        env.set_arm_target(q_target)

        dist_ee_block = np.linalg.norm(env.endeffector_position - env.block_position)
        if dist_ee_block < 0.15:
            env.set_gripper(0.0)
        else:
            env.set_gripper(1.0)

        env.step()

        q_after = env.arm_joint_positions.copy()
        ee_after = env.endeffector_position.copy()

        print(f"\n步骤 {step}:")
        print(f"  dq: {dq}")
        print(f"  q_target: {q_target}")
        print(f"  q_before: {q_before}")
        print(f"  q_after: {q_after}")
        print(f"  q变化: {q_after - q_before}")
        print(f"  ee_before: {ee_before}")
        print(f"  ee_after: {ee_after}")
        print(f"  ee变化: {ee_after - ee_before}")
        print(f"  距离目标: {env.distance_to_target}")


if __name__ == "__main__":
    main()
