"""
检查动作多样性 - 分析不同时间步的动作差异。

Usage::
    python check_action_diversity.py --data_dir data/20260626_005525
"""

import argparse
from pathlib import Path

import numpy as np


def check_action_diversity(data_dir: Path):
    """检查动作在不同时间步的多样性"""

    episode_files = sorted(data_dir.glob("episode_*.npz"))[:10]
    if not episode_files:
        print("未找到episode数据文件")
        return

    print("分析不同时间步的动作多样性:")
    print("=" * 70)

    # 检查不同时间步
    timesteps = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]

    for t in timesteps:
        actions_at_t = []
        for ep_file in episode_files:
            data = np.load(ep_file)
            act = data["actions"]
            if t < len(act):
                actions_at_t.append(act[t])

        if actions_at_t:
            actions_at_t = np.array(actions_at_t)
            std = actions_at_t.std(axis=0)
            print(f"\n时间步 t={t}:")
            print(f"  动作均值: {actions_at_t.mean(axis=0)}")
            print(f"  动作标准差: {std}")
            print(f"  标准差均值: {std.mean():.6f}")

    # 分析整个轨迹
    print("\n" + "=" * 70)
    print("轨迹分析:")
    print("=" * 70)

    # 取第一个episode的完整轨迹
    data = np.load(episode_files[0])
    act = data["actions"]
    obs = data["observations"]

    print(f"\nEpisode 0 完整轨迹:")
    print(f"  动作形状: {act.shape}")
    print(f"  动作范围: {act.min(axis=0)} ~ {act.max(axis=0)}")
    print(f"  动作标准差: {act.std(axis=0)}")

    # 检查方块位置变化是否影响动作
    print("\n" + "=" * 70)
    print("方块位置 vs 动作关系:")
    print("=" * 70)

    for i, ep_file in enumerate(episode_files[:3]):
        data = np.load(ep_file)
        obs = data["observations"]
        act = data["actions"]

        block_pos = obs[0, 11:14]
        # 找到接近方块的时间步（当ee接近方块时）
        ee_pos = obs[:, 8:11]
        dist_to_block = np.linalg.norm(ee_pos - block_pos, axis=1)

        # 找到最近的10个时间步
        closest_idx = np.argsort(dist_to_block)[:10]
        actions_near_block = act[closest_idx]

        print(f"\nEpisode {i}:")
        print(f"  方块位置: {block_pos}")
        print(f"  接近方块时的动作标准差: {actions_near_block.std(axis=0).mean():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Check action diversity across timesteps.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing collected data.",
    )
    args = parser.parse_args()

    check_action_diversity(Path(args.data_dir))


if __name__ == "__main__":
    main()
