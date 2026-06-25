"""
评估脚本 - 在仿真环境中测试训练好的行为克隆模型。

Usage::
    python evaluate.py --model_path checkpoints/model_best.pt --episodes 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv


def load_model(model_path: Path, obs_dim: int, act_dim: int, device: str) -> torch.nn.Module:
    """加载训练好的MLP模型。"""
    # 定义模型结构（需要与train.py中的模型结构一致）
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, act_dim),
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_one_episode(
    env: FrankaEnv,
    model: torch.nn.Module,
    device: str,
    max_steps: int = 3000,
    render: bool = True,
) -> Dict[str, float]:
    """运行一个评估 episode，返回评估指标。"""
    env.reset()

    total_reward = 0.0
    success = False
    steps = 0

    for step in range(max_steps):
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

        # 模型预测动作
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            action = model(obs_tensor).squeeze(0).cpu().numpy()

        # 执行动作
        arm_action = action[:4]
        grip_action = action[4] if len(action) > 4 else 0.0

        env.set_arm_target(arm_action)
        env.set_gripper(grip_action)
        env.step()

        # 检查是否成功
        if env.distance_to_target < 0.05 and finger_open < 0.02:
            success = True
            break

        steps += 1

    return {
        "success": float(success),
        "steps": steps,
        "final_distance": env.distance_to_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained behavior cloning model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
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
        default=3000,
        help="Maximum steps per episode (default: 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 初始化环境
    env = FrankaEnv(render_mode="human")

    # 观测和动作维度
    obs_dim = 20  # arm_q(4) + arm_dq(4) + ee_pos(3) + blk_pos(3) + tgt_pos(3) + gripper(3)
    act_dim = 5   # arm_target(4) + grip(1)

    # 加载模型
    model = load_model(Path(args.model_path), obs_dim, act_dim, args.device)
    print(f"Loaded model from {args.model_path}")

    # 评估
    results = []
    success_count = 0

    print(f"\nRunning {args.episodes} evaluation episodes...")
    for ep in range(args.episodes):
        result = evaluate_one_episode(
            env, model, args.device, args.max_steps, render=True
        )
        results.append(result)

        if result["success"] > 0.5:
            success_count += 1
            status = "✓"
        else:
            status = "✗"

        print(
            f"  [{ep+1:4d}/{args.episodes}] {status}  "
            f"steps={result['steps']:4d}  "
            f"dist={result['final_distance']:.4f}",
            flush=True,
        )

    # 汇总结果
    success_rate = success_count / args.episodes
    avg_steps = np.mean([r["steps"] for r in results])
    avg_dist = np.mean([r["final_distance"] for r in results])

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Success rate:     {success_rate:.2%} ({success_count}/{args.episodes})")
    print(f"  Avg steps:        {avg_steps:.1f}")
    print(f"  Avg final dist:   {avg_dist:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
