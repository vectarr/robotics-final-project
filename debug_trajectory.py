"""
调试轨迹 - 对比训练和评估时的动作。

Usage::
    python debug_trajectory.py --data_dir data/20260626_005525 --model_path data/20260626_005525/checkpoints/model_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv


def debug_trajectory(data_dir: Path, model_path: Path):
    """对比训练和评估时的动作"""

    # 加载一个训练episode
    episode_files = sorted(data_dir.glob("episode_*.npz"))
    if not episode_files:
        print("未找到episode数据文件")
        return

    data = np.load(episode_files[0])
    train_obs = data["observations"]
    train_act = data["actions"]

    print("训练数据轨迹分析:")
    print("=" * 70)
    print(f"轨迹长度: {len(train_obs)}")
    print(f"观测形状: {train_obs.shape}")
    print(f"动作形状: {train_act.shape}")

    # 分析训练数据中的关键时间点
    print("\n关键时间点的动作:")
    key_points = [0, 800, 1600, 2200, 3000, 4500, 5100, 5599]
    stage_names = ["APPROACH", "DESCEND", "GRASP", "LIFT", "MOVE", "PLACE", "RETREAT", "DONE"]

    for t, name in zip(key_points, stage_names):
        if t < len(train_act):
            print(f"  t={t:4d} ({name:8s}): {train_act[t]}")

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

    # 用模型预测训练数据的动作
    print("\n" + "=" * 70)
    print("模型预测 vs 真实动作:")
    print("=" * 70)

    for t, name in zip(key_points, stage_names):
        if t < len(train_obs):
            obs_tensor = torch.from_numpy(train_obs[t:t+1]).float()
            with torch.no_grad():
                pred_act = model(obs_tensor).squeeze(0).numpy()

            print(f"\n  t={t:4d} ({name:8s}):")
            print(f"    真实: {train_act[t]}")
            print(f"    预测: {pred_act}")
            print(f"    误差: {np.abs(pred_act - train_act[t])}")

    # 分析阶段转换
    print("\n" + "=" * 70)
    print("阶段转换分析:")
    print("=" * 70)

    # 检查夹爪状态变化
    grip_actions = train_act[:, 4]
    changes = np.where(np.abs(np.diff(grip_actions)) > 0.5)[0]

    print(f"夹爪状态变化点:")
    for t in changes:
        print(f"  t={t}: {grip_actions[t]:.4f} -> {grip_actions[t+1]:.4f}")

    # 分析位置变化
    print("\n位置变化分析:")
    ee_positions = train_obs[:, 8:11]
    block_positions = train_obs[:, 11:14]

    print(f"  初始末端位置: {ee_positions[0]}")
    print(f"  初始方块位置: {block_positions[0]}")
    print(f"  初始距离: {np.linalg.norm(ee_positions[0] - block_positions[0]):.4f}")

    # 找到最近方块的时间点
    dists = np.linalg.norm(ee_positions - block_positions, axis=1)
    closest_t = np.argmin(dists)
    print(f"\n  最近方块时间点: t={closest_t}")
    print(f"    末端位置: {ee_positions[closest_t]}")
    print(f"    方块位置: {block_positions[closest_t]}")
    print(f"    距离: {dists[closest_t]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug trajectory comparison.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing collected data.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model.",
    )
    args = parser.parse_args()

    debug_trajectory(Path(args.data_dir), Path(args.model_path))


if __name__ == "__main__":
    main()
