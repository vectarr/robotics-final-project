"""
检查夹爪动作分布。

Usage::
    python check_gripper.py --data_dir data/20260626_005525
"""

import argparse
from pathlib import Path

import numpy as np


def check_gripper_distribution(data_dir: Path):
    """检查夹爪动作分布"""

    episode_files = sorted(data_dir.glob("episode_*.npz"))[:10]
    if not episode_files:
        print("未找到episode数据文件")
        return

    print("夹爪动作分布分析:")
    print("=" * 70)

    all_actions = []
    all_observations = []
    for ep_file in episode_files:
        data = np.load(ep_file)
        act = data["actions"]
        obs = data["observations"]
        all_actions.append(act)
        all_observations.append(obs)

    all_actions = np.concatenate(all_actions, axis=0)
    all_observations = np.concatenate(all_observations, axis=0)

    # 夹爪动作分布
    grip_actions = all_actions[:, 4]
    print(f"\n夹爪动作统计:")
    print(f"  最小值: {grip_actions.min():.4f}")
    print(f"  最大值: {grip_actions.max():.4f}")
    print(f"  均值: {grip_actions.mean():.4f}")
    print(f"  标准差: {grip_actions.std():.4f}")

    # 夹爪动作分布
    open_count = np.sum(grip_actions > 0.5)
    close_count = np.sum(grip_actions <= 0.5)
    print(f"\n夹爪动作分布:")
    print(f"  打开 (>0.5): {open_count} ({open_count/len(grip_actions)*100:.1f}%)")
    print(f"  关闭 (≤0.5): {close_count} ({close_count/len(grip_actions)*100:.1f}%)")

    # 检查观测中的夹爪状态
    finger_open = all_observations[:, 17]  # finger_open
    is_closed = all_observations[:, 19]  # is_closed

    print(f"\n观测中的夹爪状态:")
    print(f"  finger_open 均值: {finger_open.mean():.4f}")
    print(f"  is_closed 均值: {is_closed.mean():.4f}")

    # 检查一个episode的夹爪动作变化
    print("\n" + "=" * 70)
    print("单个episode的夹爪动作变化:")
    print("=" * 70)

    data = np.load(episode_files[0])
    act = data["actions"]
    grip_episode = act[:, 4]

    # 找到夹爪状态变化的点
    changes = np.where(np.abs(np.diff(grip_episode)) > 0.1)[0]
    print(f"\n夹爪状态变化点 (前10个):")
    for i, t in enumerate(changes[:10]):
        print(f"  t={t}: {grip_episode[t]:.4f} -> {grip_episode[t+1]:.4f}")

    # 检查夹爪动作与距离的关系
    print("\n" + "=" * 70)
    print("夹爪动作与方块距离的关系:")
    print("=" * 70)

    for i, ep_file in enumerate(episode_files[:3]):
        data = np.load(ep_file)
        act = data["actions"]
        obs = data["observations"]

        grip = act[:, 4]
        dist_ee_block = obs[:, 17]  # dist_ee_block

        # 找到夹爪打开和关闭时的平均距离
        open_mask = grip > 0.5
        close_mask = grip <= 0.5

        if open_mask.sum() > 0 and close_mask.sum() > 0:
            avg_dist_open = dist_ee_block[open_mask].mean()
            avg_dist_close = dist_ee_block[close_mask].mean()
            print(f"\nEpisode {i}:")
            print(f"  夹爪打开时平均距离: {avg_dist_open:.4f}")
            print(f"  夹爪关闭时平均距离: {avg_dist_close:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Check gripper action distribution.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing collected data.",
    )
    args = parser.parse_args()

    check_gripper_distribution(Path(args.data_dir))


if __name__ == "__main__":
    main()
