"""
检查采集的数据质量 - 用于诊断数据问题。

Usage::
    python check_data.py --data_dir data/20260626_004329
"""

import argparse
import json
from pathlib import Path

import numpy as np


def check_data_quality(data_dir: Path):
    """检查数据质量和多样性"""

    # 加载实验数据
    experiment_file = data_dir / "experiment_data.json"
    if experiment_file.exists():
        with open(experiment_file, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)
        episodes = experiment_data["episodes"]
        print(f"总episodes: {len(episodes)}")
        print(f"成功率: {experiment_data['summary']['success_rate']:.2%}")
    else:
        print("未找到experiment_data.json")
        return

    # 检查几个episode的数据
    episode_files = sorted(data_dir.glob("episode_*.npz"))[:5]
    if not episode_files:
        print("未找到episode数据文件")
        return

    print("\n检查前5个episode的数据:")
    print("-" * 60)

    block_positions = []
    target_positions = []
    first_actions = []

    for ep_file in episode_files:
        data = np.load(ep_file)
        obs = data["observations"]
        act = data["actions"]

        # 提取方块位置和目标位置（从第一帧观测）
        block_pos = obs[0, 11:14]  # blk_pos
        target_pos = obs[0, 14:17]  # tgt_pos
        first_action = act[0]

        block_positions.append(block_pos)
        target_positions.append(target_pos)
        first_actions.append(first_action)

        print(f"{ep_file.name}:")
        print(f"  方块位置: {block_pos}")
        print(f"  目标位置: {target_pos}")
        print(f"  首帧动作: {first_action}")
        print(f"  数据形状: obs={obs.shape}, act={act.shape}")
        print()

    # 检查多样性
    block_positions = np.array(block_positions)
    target_positions = np.array(target_positions)
    first_actions = np.array(first_actions)

    print("=" * 60)
    print("数据多样性分析:")
    print("-" * 60)

    print(f"\n方块位置变化:")
    print(f"  范围: {block_positions.min(axis=0)} ~ {block_positions.max(axis=0)}")
    print(f"  标准差: {block_positions.std(axis=0)}")

    print(f"\n目标位置变化:")
    print(f"  范围: {target_positions.min(axis=0)} ~ {target_positions.max(axis=0)}")
    print(f"  标准差: {target_positions.std(axis=0)}")

    print(f"\n首帧动作变化:")
    print(f"  范围: {first_actions.min(axis=0)} ~ {first_actions.max(axis=0)}")
    print(f"  标准差: {first_actions.std(axis=0)}")

    # 判断是否有足够的多样性
    block_std = block_positions.std(axis=0).mean()
    action_std = first_actions.std(axis=0).mean()

    print("\n" + "=" * 60)
    print("诊断结果:")
    print("-" * 60)

    if block_std < 0.001:
        print("⚠️  方块位置几乎没有变化！")
        print("   可能原因: 随机化没有生效，或者随机化范围太小")
    else:
        print(f"✓  方块位置有变化 (std={block_std:.4f})")

    if action_std < 0.001:
        print("⚠️  动作几乎完全一样！")
        print("   可能原因: 方块位置没有变化，或者控制器输出固定")
    else:
        print(f"✓  动作有多样性 (std={action_std:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Check data quality and diversity.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing collected data.",
    )
    args = parser.parse_args()

    check_data_quality(Path(args.data_dir))


if __name__ == "__main__":
    main()
