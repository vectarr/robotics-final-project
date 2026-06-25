"""
检查训练数据的动作范围和关节限制。

Usage::
    python check_action_range.py --data_dir data/20260626_005525
"""

import argparse
from pathlib import Path

import numpy as np


def check_action_range(data_dir: Path):
    """检查训练数据的动作范围"""

    episode_files = sorted(data_dir.glob("episode_*.npz"))[:10]
    if not episode_files:
        print("未找到episode数据文件")
        return

    print("检查训练数据动作范围:")
    print("=" * 70)

    all_actions = []
    for ep_file in episode_files:
        data = np.load(ep_file)
        act = data["actions"]
        all_actions.append(act)

    all_actions = np.concatenate(all_actions, axis=0)

    print(f"样本数量: {len(all_actions)}")
    print(f"\n动作范围:")
    print(f"  关节1: [{all_actions[:, 0].min():.4f}, {all_actions[:, 0].max():.4f}]")
    print(f"  关节2: [{all_actions[:, 1].min():.4f}, {all_actions[:, 1].max():.4f}]")
    print(f"  关节3: [{all_actions[:, 2].min():.4f}, {all_actions[:, 2].max():.4f}]")
    print(f"  关节4: [{all_actions[:, 3].min():.4f}, {all_actions[:, 3].max():.4f}]")
    print(f"  夹爪:  [{all_actions[:, 4].min():.4f}, {all_actions[:, 4].max():.4f}]")

    print(f"\n动作统计:")
    print(f"  均值: {all_actions.mean(axis=0)}")
    print(f"  标准差: {all_actions.std(axis=0)}")

    # 检查关节限制（从MuJoCo模型）
    print("\n" + "=" * 70)
    print("关节限制（从IK控制器推断）:")
    print("=" * 70)

    # IK控制器中的关节限制代码
    print("  关节限制在ik_controller.py中定义:")
    print("  for i in range(4):")
    print("      jnt_id = self.env._arm_joint_ids[i]")
    print("      lo = self.model.jnt_range[jnt_id, 0]")
    print("      hi = self.model.jnt_range[jnt_id, 1]")
    print("      q_new[i] = np.clip(q_new[i], lo, hi)")

    # 检查是否有超出范围的动作
    print("\n" + "=" * 70)
    print("潜在问题:")
    print("=" * 70)

    # 常见的关节范围
    typical_ranges = [
        (-2.8973, 2.8973),  # joint1
        (-1.7628, 1.7628),  # joint2
        (-2.8973, 2.8973),  # joint3 (joint4 in our case)
        (-3.0718, -0.0698),  # joint4 (joint6 in our case)
    ]

    for i, (lo, hi) in enumerate(typical_ranges):
        out_of_range = np.sum((all_actions[:, i] < lo) | (all_actions[:, i] > hi))
        if out_of_range > 0:
            print(f"  ⚠️  关节{i+1}: {out_of_range}个动作超出范围 [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"  ✓  关节{i+1}: 所有动作在范围内 [{lo:.4f}, {hi:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Check action range in training data.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing collected data.",
    )
    args = parser.parse_args()

    check_action_range(Path(args.data_dir))


if __name__ == "__main__":
    main()
