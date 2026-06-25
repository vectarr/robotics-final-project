"""
Data collection for behavioural cloning — IK-controller expert trajectories.

Collects (observation, action) pairs from the IK pick-and-place controller
across many episodes with randomised block positions.  The dataset is saved
as compressed ``.npz`` files, one per episode, plus a metadata summary.

Usage::

    python collect.py                                    # 100 episodes, fixed block
    python collect.py --rand                             # randomise block position
    python collect.py --episodes 500 --rand              # 500 episodes
    python collect.py --episodes 200 --out data/my_run   # custom output dir

Output layout::

    data/
    ├── metadata.npz          # episode index, success flags, block/target pos
    ├── episode_0000.npz      # observations, actions
    ├── episode_0001.npz
    └── ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 添加项目根目录到Python路径
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 确保env模块可以被找到
from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv
from controllers.ik_controller import IKController, MAX_STEPS as IK_MAX_STEPS


class Logger:
    """同时输出到终端和日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class ExperimentRecorder:
    """实验数据记录器，用于保存详细的实验数据供后续分析和报告使用"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.experiment_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "episodes": [],
            "summary": {}
        }

    def save_config(self, args):
        """保存实验配置"""
        self.experiment_data["config"] = {
            "episodes": args.episodes,
            "randomize_block": args.rand,
            "seed": args.seed,
            "output_dir": str(args.out),
            "max_steps": IK_MAX_STEPS,
        }

    def record_episode(self, ep_idx: int, success: bool, steps: int,
                       block_pos: np.ndarray, target_pos: np.ndarray,
                       episode_time: float):
        """记录单个episode的数据"""
        episode_data = {
            "episode": ep_idx,
            "success": bool(success),
            "steps": int(steps),
            "block_position": block_pos.tolist() if block_pos is not None else None,
            "target_position": target_pos.tolist() if target_pos is not None else None,
            "duration_seconds": round(episode_time, 3),
        }
        self.experiment_data["episodes"].append(episode_data)

    def save_summary(self, success_count: int, fail_count: int,
                     total_steps: int, elapsed: float):
        """保存实验总结"""
        self.experiment_data["summary"] = {
            "total_episodes": success_count + fail_count,
            "successful_episodes": success_count,
            "failed_episodes": fail_count,
            "success_rate": round(success_count / max(success_count + fail_count, 1), 4),
            "total_steps": total_steps,
            "avg_steps_per_episode": round(total_steps / max(success_count, 1), 1),
            "total_time_seconds": round(elapsed, 2),
            "episodes_per_second": round(success_count / max(elapsed, 0.01), 2),
            "end_time": datetime.now().isoformat(),
        }

    def save_to_json(self):
        """保存完整的实验数据到JSON文件"""
        json_path = self.output_dir / "experiment_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
        print(f"实验数据已保存到: {json_path}")

    def save_episode_statistics(self):
        """保存episode统计信息，方便后续绘图"""
        episodes = self.experiment_data["episodes"]
        if not episodes:
            return

        # 提取统计数据
        successes = [ep["success"] for ep in episodes]
        steps = [ep["steps"] for ep in episodes]
        durations = [ep["duration_seconds"] for ep in episodes]

        # 计算累积成功率
        cumulative_success = []
        success_count = 0
        for i, success in enumerate(successes):
            success_count += int(success)
            cumulative_success.append(success_count / (i + 1))

        stats = {
            "episode_indices": list(range(len(episodes))),
            "successes": successes,
            "steps": steps,
            "durations": durations,
            "cumulative_success_rate": cumulative_success,
        }

        stats_path = self.output_dir / "episode_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"统计数据已保存到: {stats_path}")


def _collect_one_episode(
    env: FrankaEnv,
    controller: IKController,
) -> Optional[Dict[str, np.ndarray]]:
    """Run one pick-and-place episode and return (obs, act) arrays.

    Returns ``None`` if the episode times out (failure).
    """
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    for _ in range(IK_MAX_STEPS):
        # ---- build observation (before action) ---------------------------
        """
        下面给出示例观测变量 obs
        可以自行按需拓展
        """
        arm_q = env.arm_joint_positions  # (4,)
        arm_dq = env.arm_joint_velocities  # (4,)
        ee_pos = env.endeffector_position  # (3,)
        blk_pos = env.block_position  # (3,)
        tgt_pos = env.target_position  # (3,)
        finger_pos = env.finger_joint_positions  # (2,)
        # 夹爪开度
        finger_open = float(np.mean(finger_pos))
        # 末端执行器到方块的距离
        dist_ee_block = float(np.linalg.norm(ee_pos - blk_pos))
        # < 0.02 认为夹爪闭合
        is_closed = 1.0 if finger_open < 0.02 else 0.0

        obs = np.concatenate(
            [
                arm_q,
                arm_dq,
                ee_pos,
                blk_pos,
                tgt_pos,
                [finger_open, dist_ee_block, is_closed],
            ]
        )
        obs_list.append(obs.astype(np.float32))

        # ---- compute control (sets ctrl internally) ---------------------
        controller.compute_control()

        # ---- build action (the control just computed) -------------------
        arm_target = np.array(
            [env.data.ctrl[aid] for aid in env._arm_actuator_ids],
            dtype=np.float32,
        )
        grip_cmd = float(env.data.ctrl[env._finger_actuator_id]) / 255.0
        act = np.concatenate([arm_target, [grip_cmd]])
        act_list.append(act.astype(np.float32))

        # ---- step simulation --------------------------------------------
        env.step()

        if controller.is_done():
            break
    else:
        # Timed out — episode is a failure
        return None

    return {
        "observations": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect IK-controller trajectories for behavioural cloning.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: PROJECT_ROOT/data/<timestamp>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--rand",
        action="store_true",
        default=False,
        help="Randomise block initial position (default: off).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- output directory ------------------------------------------------
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(PROJECT_ROOT_DIR) / "data" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- setup logger ----------------------------------------------------
    log_file = out_dir / "collect.log"
    sys.stdout = Logger(log_file)

    # ---- setup recorder --------------------------------------------------
    recorder = ExperimentRecorder(out_dir)
    recorder.save_config(args)

    print(f"Log file: {log_file}")
    print(f"Output directory: {out_dir}")

    # ---- environment & controller ---------------------------------------
    env = FrankaEnv(randomize_block=args.rand)

    # ---- collection loop -------------------------------------------------
    success_count = 0
    fail_count = 0
    total_steps = 0
    start_time = time.perf_counter()

    metadata: Dict[str, list] = {
        "episode": [],
        "success": [],
        "steps": [],
        "block_init_pos": [],
        "target_pos": [],
    }

    for ep in range(args.episodes):
        ep_start_time = time.perf_counter()
        env.reset()
        controller = IKController(env)

        data = _collect_one_episode(env, controller)
        ep_elapsed = time.perf_counter() - ep_start_time

        if data is None:
            fail_count += 1
            status = "✗"
            recorder.record_episode(ep, False, 0, None, None, ep_elapsed)
        else:
            success_count += 1
            total_steps += len(data["observations"])

            # Save episode
            ep_path = out_dir / f"episode_{ep:04d}.npz"
            np.savez_compressed(ep_path, **data)

            # Track metadata
            block_pos = data["observations"][0, 11:14]
            target_pos = data["observations"][0, 14:17]
            metadata["episode"].append(ep)
            metadata["success"].append(1)
            metadata["steps"].append(len(data["observations"]))
            metadata["block_init_pos"].append(block_pos)
            metadata["target_pos"].append(target_pos)
            status = "✓"

            # Record episode data
            recorder.record_episode(ep, True, len(data["observations"]),
                                   block_pos, target_pos, ep_elapsed)

        elapsed = time.perf_counter() - start_time
        rate = (ep + 1) / elapsed if elapsed > 0 else 0.0
        n_steps = len(data["observations"]) if data else 0
        print(
            f"  [{ep+1:4d}/{args.episodes}] {status}  "
            f"steps={n_steps:4d}  "
            f"success={success_count}  fail={fail_count}  "
            f"{rate:.1f} ep/s  "
            f"time={ep_elapsed:.1f}s",
            flush=True,
        )

    # ---- save metadata ---------------------------------------------------
    elapsed = time.perf_counter() - start_time
    meta_path = out_dir / "metadata.npz"
    np.savez_compressed(
        meta_path,
        episode=np.array(metadata["episode"], dtype=np.int32),
        success=np.array(metadata["success"], dtype=np.int32),
        steps=np.array(metadata["steps"], dtype=np.int32),
        block_init_pos=np.array(metadata["block_init_pos"], dtype=np.float32),
        target_pos=np.array(metadata["target_pos"], dtype=np.float32),
    )

    # ---- save experiment data --------------------------------------------
    recorder.save_summary(success_count, fail_count, total_steps, elapsed)
    recorder.save_to_json()
    recorder.save_episode_statistics()

    # ---- summary ---------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Collection complete")
    print(f"  ─────────────────")
    print(f"  Episodes requested:  {args.episodes}")
    print(f"  Successful:          {success_count}")
    print(f"  Failed (timeout):    {fail_count}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Avg steps/episode:   {total_steps / max(success_count, 1):.0f}")
    print(f"  Wall-clock time:     {elapsed:.1f} s")
    print(f"  Episodes/second:     {success_count / elapsed:.2f}")
    print(f"  Output directory:    {out_dir}")
    print("=" * 60)
    print()
    print("生成的文件:")
    print(f"  - collect.log: 完整的运行日志")
    print(f"  - experiment_data.json: 详细的实验数据")
    print(f"  - episode_statistics.json: 统计数据（用于绘图）")
    print(f"  - metadata.npz: 元数据")
    print(f"  - episode_*.npz: 每个episode的数据")


if __name__ == "__main__":
    main()
