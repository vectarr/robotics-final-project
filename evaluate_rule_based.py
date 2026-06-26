"""
基于规则的评估脚本 - 使用简单规则控制夹爪。

Usage::
    python evaluate_rule_based.py --model_path data/20260626_005525/checkpoints/model_best.pt --episodes 10
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv


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


def load_model(model_path: Path, obs_dim: int, act_dim: int, device: str) -> torch.nn.Module:
    """加载训练好的MLP模型。"""
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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_one_episode(
    env: FrankaEnv,
    model: torch.nn.Module,
    device: str,
    max_steps: int = 5600,
    render: bool = True,
) -> Dict[str, float]:
    """运行一个评估 episode，使用规则控制夹爪。"""
    env.reset()

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

        # 模型预测关节动作
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            pred_act = model(obs_tensor).squeeze(0).cpu().numpy()

        # 使用规则控制夹爪（基于训练数据的模式）
        # 规则：距离<0.15时关闭，距离>0.2时打开
        if dist_ee_block < 0.15:
            grip_action = 0.0  # 关闭
        elif dist_ee_block > 0.2:
            grip_action = 1.0  # 打开
        else:
            # 中间区域，保持当前状态
            grip_action = 1.0 if finger_open > 0.02 else 0.0

        # 应用动作
        arm_action = pred_act[:4]
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
        description="Evaluate with rule-based gripper control.",
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
        default=5600,
        help="Maximum steps per episode (default: 5600).",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output directory
    model_path = Path(args.model_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = output_dir / "evaluate_rule_based.log"
    sys.stdout = Logger(log_file)

    # 初始化环境
    env = FrankaEnv(render_mode="human")

    # 观测和动作维度
    obs_dim = 20
    act_dim = 5

    # 加载模型
    model = load_model(model_path, obs_dim, act_dim, args.device)
    print(f"Loaded model from {model_path}")
    print(f"Output directory: {output_dir}")

    # 评估
    results = []
    success_count = 0

    print(f"\nRunning {args.episodes} evaluation episodes with rule-based gripper...")
    for ep in range(args.episodes):
        ep_start = time.perf_counter()
        result = evaluate_one_episode(
            env, model, args.device, args.max_steps, render=True
        )
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
    print("Evaluation Results (Rule-based Gripper)")
    print("=" * 60)
    print(f"  Success rate:     {success_rate:.2%} ({success_count}/{args.episodes})")
    print(f"  Avg steps:        {avg_steps:.1f}")
    print(f"  Avg final dist:   {avg_dist:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
