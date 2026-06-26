"""
检查训练数据格式和模型预测。

Usage::
    python check_training_data.py --data_dir data/20260626_005525 --model_path data/20260626_005525/checkpoints/model_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def check_training_data(data_dir: Path, model_path: Path):
    """检查训练数据和模型预测"""

    # 加载数据
    episode_files = sorted(data_dir.glob("episode_*.npz"))[:5]
    if not episode_files:
        print("未找到episode数据文件")
        return

    print("检查训练数据格式:")
    print("=" * 70)

    for ep_file in episode_files:
        data = np.load(ep_file)
        obs = data["observations"]
        act = data["actions"]

        print(f"\n{ep_file.name}:")
        print(f"  观测形状: {obs.shape}")
        print(f"  动作形状: {act.shape}")

        # 检查观测的各个部分
        print(f"\n  观测内容:")
        print(f"    arm_q (关节角度): {obs[0, :4]}")
        print(f"    arm_dq (关节速度): {obs[0, 4:8]}")
        print(f"    ee_pos (末端位置): {obs[0, 8:11]}")
        print(f"    blk_pos (方块位置): {obs[0, 11:14]}")
        print(f"    tgt_pos (目标位置): {obs[0, 14:17]}")
        print(f"    finger_open (夹爪开度): {obs[0, 17]}")
        print(f"    dist_ee_block (距离): {obs[0, 18]}")
        print(f"    is_closed (是否闭合): {obs[0, 19]}")

        # 检查动作
        print(f"\n  动作内容:")
        print(f"    arm_target (关节目标): {act[0, :4]}")
        print(f"    grip_cmd (夹爪命令): {act[0, 4]}")

    # 检查模型预测
    if model_path.exists():
        print("\n" + "=" * 70)
        print("检查模型预测:")
        print("=" * 70)

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

        # 在第一个episode上测试
        data = np.load(episode_files[0])
        obs = data["observations"]
        act = data["actions"]

        # 随机采样几个时间步
        np.random.seed(42)
        indices = np.random.choice(len(obs), 5, replace=False)

        for idx in indices:
            obs_tensor = torch.from_numpy(obs[idx:idx+1]).float()
            with torch.no_grad():
                pred_act = model(obs_tensor).squeeze(0).numpy()

            print(f"\n时间步 {idx}:")
            print(f"  真实动作: {act[idx]}")
            print(f"  预测动作: {pred_act}")
            print(f"  误差: {np.abs(pred_act - act[idx])}")

        # 检查整个轨迹的预测
        print("\n" + "=" * 70)
        print("轨迹预测分析:")
        print("=" * 70)

        obs_tensor = torch.from_numpy(obs).float()
        with torch.no_grad():
            pred_acts = model(obs_tensor).numpy()

        print(f"  真实动作范围: {act.min(axis=0)} ~ {act.max(axis=0)}")
        print(f"  预测动作范围: {pred_acts.min(axis=0)} ~ {pred_acts.max(axis=0)}")

        # 检查夹爪预测
        print(f"\n  夹爪预测:")
        print(f"    真实夹爪范围: [{act[:, 4].min():.4f}, {act[:, 4].max():.4f}]")
        print(f"    预测夹爪范围: [{pred_acts[:, 4].min():.4f}, {pred_acts[:, 4].max():.4f}]")
        print(f"    真实夹爪均值: {act[:, 4].mean():.4f}")
        print(f"    预测夹爪均值: {pred_acts[:, 4].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Check training data and model predictions.",
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
        default=None,
        help="Path to trained model.",
    )
    args = parser.parse_args()

    check_training_data(Path(args.data_dir), Path(args.model_path) if args.model_path else None)


if __name__ == "__main__":
    main()
