"""
调试评估环境 - 检查环境状态和动作应用。

Usage::
    python debug_eval_env.py --model_path data/20260626_005525/checkpoints/model_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv


def debug_eval_env(model_path: Path):
    """调试评估环境"""

    # 初始化环境
    env = FrankaEnv(render_mode="human")

    print("环境初始化:")
    print("=" * 70)

    # 重置环境
    env.reset()

    print(f"初始状态:")
    print(f"  关节角度: {env.arm_joint_positions}")
    print(f"  关节速度: {env.arm_joint_velocities}")
    print(f"  末端位置: {env.endeffector_position}")
    print(f"  方块位置: {env.block_position}")
    print(f"  目标位置: {env.target_position}")
    print(f"  夹爪位置: {env.finger_joint_positions}")
    print(f"  距离目标: {env.distance_to_target}")

    # 加载模型
    obs_dim = 20
    act_dim = 5
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, act_dim),
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    print("\n" + "=" * 70)
    print("运行10步，观察动作:")
    print("=" * 70)

    for step in range(10):
        # 构建观测
        arm_q = env.arm_joint_positions
        arm_dq = env.arm_joint_velocities
        ee_pos = env.endeffector_position
        blk_pos = env.block_position
        tgt_pos = env.target_position
        finger_pos = env.finger_joint_positions

        finger_open = float(np.mean(finger_pos))
        dist_ee_block = float(np.linalg.norm(ee_pos - blk_pos))
        is_closed = 1.0 if finger_open < 0.02 else 0.0

        obs = np.concatenate([
            arm_q, arm_dq, ee_pos, blk_pos, tgt_pos,
            [finger_open, dist_ee_block, is_closed],
        ]).astype(np.float32)

        # 模型预测
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.no_grad():
            pred_act = model(obs_tensor).squeeze(0).numpy()

        print(f"\n步骤 {step}:")
        print(f"  观测:")
        print(f"    关节角度: {arm_q}")
        print(f"    末端位置: {ee_pos}")
        print(f"    方块位置: {blk_pos}")
        print(f"    距离方块: {dist_ee_block:.4f}")
        print(f"  预测动作: {pred_act}")

        # 应用动作
        arm_action = pred_act[:4]
        grip_action = pred_act[4]

        print(f"  应用动作:")
        print(f"    关节目标: {arm_action}")
        print(f"    夹爪命令: {grip_action:.4f}")

        env.set_arm_target(arm_action)
        env.set_gripper(grip_action)
        env.step()

        print(f"  执行后:")
        print(f"    关节角度: {env.arm_joint_positions}")
        print(f"    末端位置: {env.endeffector_position}")
        print(f"    ctrl[关节]: {[env.data.ctrl[aid] for aid in env._arm_actuator_ids]}")
        print(f"    ctrl[夹爪]: {env.data.ctrl[env._finger_actuator_id]:.4f}")

    # 检查ctrl值
    print("\n" + "=" * 70)
    print("当前ctrl值:")
    print("=" * 70)
    for i, aid in enumerate(env._arm_actuator_ids):
        print(f"  关节{i+1} ctrl: {env.data.ctrl[aid]:.4f}")
    print(f"  夹爪 ctrl: {env.data.ctrl[env._finger_actuator_id]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug evaluation environment.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model.",
    )
    args = parser.parse_args()

    debug_eval_env(Path(args.model_path))


if __name__ == "__main__":
    main()
