"""
使用IK控制器进行评估 - 验证环境是否正常工作。

Usage::
    python evaluate_ik.py --episodes 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv
from controllers.ik_controller import IKController, MAX_STEPS


def evaluate_one_episode(
    env: FrankaEnv,
    max_steps: int = MAX_STEPS,
    render: bool = True,
) -> dict:
    """使用IK控制器运行一个评估episode。"""
    env.reset()
    controller = IKController(env)

    success = False
    steps = 0

    for step in range(max_steps):
        # 使用IK控制器计算动作
        controller.compute_control()

        # 执行动作
        env.step()

        # 检查是否成功
        if env.distance_to_target < 0.05 and float(np.mean(env.finger_joint_positions)) < 0.02:
            success = True
            break

        steps += 1

        if controller.is_done():
            break

    return {
        "success": float(success),
        "steps": steps,
        "final_distance": env.distance_to_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate with IK controller.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 初始化环境
    env = FrankaEnv(render_mode="human")

    # 评估
    results = []
    success_count = 0

    print(f"Running {args.episodes} evaluation episodes with IK controller...")
    for ep in range(args.episodes):
        ep_start = time.perf_counter()
        result = evaluate_one_episode(env, args.max_steps, render=True)
        ep_time = time.perf_counter() - ep_start
        results.append(result)

        if result["success"] > 0.5:
            success_count += 1
            status = "✓"
        else:
            status = "✗"

        print(
            f"  [{ep+1:4d}/{args.episodes}] {status}  "
            f"steps={result['steps']:4d}  "
            f"dist={result['final_distance']:.4f}  "
            f"time={ep_time:.1f}s",
            flush=True,
        )

    # 汇总结果
    success_rate = success_count / args.episodes
    avg_steps = np.mean([r["steps"] for r in results])
    avg_dist = np.mean([r["final_distance"] for r in results])

    print("\n" + "=" * 60)
    print("Evaluation Results (IK Controller)")
    print("=" * 60)
    print(f"  Success rate:     {success_rate:.2%} ({success_count}/{args.episodes})")
    print(f"  Avg steps:        {avg_steps:.1f}")
    print(f"  Avg final dist:   {avg_dist:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
