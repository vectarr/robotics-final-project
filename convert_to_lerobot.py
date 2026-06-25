"""
将collect.py采集的npz数据转换为LeRobot训练所需的zarr格式。

Usage::
    python convert_to_lerobot.py --data_dir data/20260625_123456 --output_dir data/lerobot_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import zarr


def convert_npz_to_zarr(
    data_dir: Path,
    output_dir: Path,
    chunk_size: int = 1000,
) -> None:
    """将npz格式的episode数据转换为zarr格式。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有episode数据
    episode_files = sorted(data_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")

    print(f"Found {len(episode_files)} episodes")

    # 读取所有数据
    all_observations = []
    all_actions = []
    episode_lengths = []

    for ep_file in episode_files:
        data = np.load(ep_file)
        obs = data["observations"]
        act = data["actions"]
        all_observations.append(obs)
        all_actions.append(act)
        episode_lengths.append(len(obs))

    # 拼接所有数据
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    # 创建episode index
    episode_indices = []
    for ep_idx, length in enumerate(episode_lengths):
        episode_indices.extend([ep_idx] * length)
    episode_indices = np.array(episode_indices, dtype=np.int64)

    # 创建step indices
    step_indices = []
    for length in episode_lengths:
        step_indices.extend(range(length))
    step_indices = np.array(step_indices, dtype=np.int64)

    # 保存为zarr格式
    root = zarr.open(str(output_dir), mode="w")

    # 保存数据
    root.create_dataset(
        "observations/joint_positions",
        data=observations[:, :4],  # arm_q
        chunks=(min(chunk_size, len(observations)), 4),
        overwrite=True,
    )
    root.create_dataset(
        "observations/joint_velocities",
        data=observations[:, 4:8],  # arm_dq
        chunks=(min(chunk_size, len(observations)), 4),
        overwrite=True,
    )
    root.create_dataset(
        "observations/end_effector_position",
        data=observations[:, 8:11],  # ee_pos
        chunks=(min(chunk_size, len(observations)), 3),
        overwrite=True,
    )
    root.create_dataset(
        "observations/block_position",
        data=observations[:, 11:14],  # blk_pos
        chunks=(min(chunk_size, len(observations)), 3),
        overwrite=True,
    )
    root.create_dataset(
        "observations/target_position",
        data=observations[:, 14:17],  # tgt_pos
        chunks=(min(chunk_size, len(observations)), 3),
        overwrite=True,
    )
    root.create_dataset(
        "observations/gripper_state",
        data=observations[:, 17:20],  # finger_open, dist_ee_block, is_closed
        chunks=(min(chunk_size, len(observations)), 3),
        overwrite=True,
    )
    root.create_dataset(
        "actions",
        data=actions,
        chunks=(min(chunk_size, len(observations)), actions.shape[1]),
        overwrite=True,
    )

    # 保存索引
    root.create_dataset("episode_index", data=episode_indices, overwrite=True)
    root.create_dataset("step_index", data=step_indices, overwrite=True)

    # 保存元数据
    root.attrs["num_episodes"] = len(episode_files)
    root.attrs["num_steps"] = len(observations)
    root.attrs["observation_shape"] = observations.shape[1:]
    root.attrs["action_shape"] = actions.shape[1:]
    root.attrs["episode_lengths"] = episode_lengths

    print(f"Converted {len(episode_files)} episodes to zarr format")
    print(f"  Total steps: {len(observations)}")
    print(f"  Observation shape: {observations.shape[1:]}")
    print(f"  Action shape: {actions.shape[1:]}")
    print(f"  Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert npz episode data to zarr format for LeRobot training.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input directory containing episode_*.npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for zarr dataset (default: data_dir/zarr_dataset).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size for zarr arrays (default: 1000).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "zarr_dataset"

    convert_npz_to_zarr(data_dir, output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
