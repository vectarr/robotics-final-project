"""
评估脚本 - 在仿真环境中测试训练好的行为克隆模型。

Usage::
    python evaluate.py --model_path checkpoints/model_best.pt --episodes 10
"""

from __future__ import annotations

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


class EvaluationRecorder:
    """评估过程记录器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.episodes = []
        self.config = {}

    def save_config(self, args, model_path):
        """保存评估配置"""
        self.config = {
            "model_path": str(model_path),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "device": args.device,
            "start_time": datetime.now().isoformat(),
        }

    def record_episode(self, ep_idx: int, success: bool, steps: int,
                       final_distance: float, episode_time: float):
        """记录单个评估episode"""
        self.episodes.append({
            "episode": ep_idx,
            "success": bool(success),
            "steps": steps,
            "final_distance": round(final_distance, 6),
            "duration_seconds": round(episode_time, 3),
        })

    def save_results(self):
        """保存评估结果"""
        successes = [ep["success"] for ep in self.episodes]
        steps = [ep["steps"] for ep in self.episodes]
        distances = [ep["final_distance"] for ep in self.episodes]

        results = {
            "config": self.config,
            "episodes": self.episodes,
            "summary": {
                "total_episodes": len(self.episodes),
                "successful_episodes": sum(successes),
                "success_rate": round(sum(successes) / max(len(successes), 1), 4),
                "avg_steps": round(np.mean(steps), 1) if steps else 0,
                "avg_final_distance": round(np.mean(distances), 4) if distances else 0,
                "min_steps": min(steps) if steps else 0,
                "max_steps": max(steps) if steps else 0,
                "end_time": datetime.now().isoformat(),
            }
        }

        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到: {results_path}")


def load_model(model_path: Path, obs_dim: int, act_dim: int, device: str) -> torch.nn.Module:
    """加载训练好的MLP模型。"""
    # 定义模型结构（需要与train.py中的模型结构一致）
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
        torch.nn.Sigmoid(),  # 输出在[0,1]范围内，适合夹爪动作
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: same as model directory).",
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
    log_file = output_dir / "evaluate.log"
    sys.stdout = Logger(log_file)

    # Setup recorder
    recorder = EvaluationRecorder(output_dir)
    recorder.save_config(args, model_path)

    # 初始化环境
    env = FrankaEnv(render_mode="human")

    # 观测和动作维度
    obs_dim = 20  # arm_q(4) + arm_dq(4) + ee_pos(3) + blk_pos(3) + tgt_pos(3) + gripper(3)
    act_dim = 5   # arm_target(4) + grip(1)

    # 加载模型
    model = load_model(model_path, obs_dim, act_dim, args.device)
    print(f"Loaded model from {model_path}")
    print(f"Output directory: {output_dir}")

    # 评估
    results = []
    success_count = 0

    print(f"\nRunning {args.episodes} evaluation episodes...")
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

        # Record episode
        recorder.record_episode(ep, result["success"] > 0.5,
                               result["steps"], result["final_distance"], ep_time)

        print(
            f"  [{ep+1:4d}/{args.episodes}] {status}  "
            f"steps={result['steps']:4d}  "
            f"dist={result['final_distance']:.4f}  "
            f"time={ep_time:.1f}s",
            flush=True,
        )

    # Save results
    recorder.save_results()

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
    print(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")
    print(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main()
