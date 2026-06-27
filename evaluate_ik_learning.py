"""
IK学习评估脚本 - 使用学习的Jacobian计算进行评估。

Usage::
    python evaluate_ik_learning.py --model_path data/20260626_005525/checkpoints_ik/model_best.pt --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv


class Logger:
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


class IKModel(torch.nn.Module):
    """IK学习模型（与训练时一致）。"""

    def __init__(self, input_dim: int = 9, output_dim: int = 4):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
            torch.nn.Tanh(),
        )

        self.output_scale = 0.02

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) * self.output_scale


def load_model(model_path: Path, input_dim: int, output_dim: int, device: str) -> IKModel:
    """加载训练好的模型。"""
    model = IKModel(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def build_features(env: FrankaEnv) -> np.ndarray:
    """构建输入特征。"""
    q_current = env.arm_joint_positions  # (4,)
    ee_pos = env.endeffector_position   # (3,)
    blk_pos = env.block_position        # (3,)
    tgt_pos = env.target_position       # (3,)
    finger_open = float(np.mean(env.finger_joint_positions))
    dist_ee_block = float(np.linalg.norm(ee_pos - blk_pos))

    # 根据阶段确定目标位置
    if dist_ee_block < 0.15 and finger_open < 0.02:
        # LIFT + MOVE + PLACE阶段：目标是目标位置
        target_pos = tgt_pos
    else:
        # APPROACH + DESCEND阶段：目标是方块位置
        target_pos = blk_pos

    # 计算位置误差
    pos_error = target_pos - ee_pos  # (3,)
    dist_to_target = np.linalg.norm(pos_error)

    # 计算阶段
    if dist_ee_block > 0.3:
        phase = 0.0  # APPROACH
    elif dist_ee_block > 0.15:
        phase = 0.2  # DESCEND
    elif finger_open < 0.02:
        # 根据z坐标区分LIFT和MOVE
        if ee_pos[2] < 0.35:
            phase = 0.4  # GRASP + LIFT (z较低)
        else:
            phase = 0.6  # MOVE (z较高)
    else:
        phase = 0.8  # PLACE

    # 特征：[q_current, pos_error, dist_to_target, phase]
    features = np.concatenate([
        q_current,
        pos_error,
        [dist_to_target],
        [phase],
    ]).astype(np.float32)

    return features


def evaluate_one_episode(
    env: FrankaEnv,
    model: IKModel,
    device: str,
    max_steps: int = 5600,
    render: bool = True,
) -> Dict[str, float]:
    """运行一个评估episode。"""
    env.reset()

    success = False
    steps = 0

    for step in range(max_steps):
        # 从环境读取当前实际关节位置
        q_current = env.arm_joint_positions.copy()

        # 构建特征
        features = build_features(env)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

        # 模型预测关节增量
        with torch.no_grad():
            dq = model(features_tensor).squeeze(0).cpu().numpy()

        # 计算关节目标
        q_target = q_current + dq

        # 关节限制
        joint_limits = [
            (-2.8973, 2.8973),   # joint1
            (-1.7628, 1.7628),   # joint2
            (-2.8973, 2.8973),   # joint3
            (-3.0718, -0.0698),  # joint4
        ]
        for i, (lo, hi) in enumerate(joint_limits):
            q_target[i] = np.clip(q_target[i], lo, hi)

        # 应用动作
        env.set_arm_target(q_target)

        # 夹爪控制（基于距离）
        dist_ee_block = np.linalg.norm(env.endeffector_position - env.block_position)
        if dist_ee_block < 0.15:
            grip_action = 0.0  # 关闭
        else:
            grip_action = 1.0  # 打开
        env.set_gripper(grip_action)

        # 执行
        env.step()

        # 检查成功
        finger_open = float(np.mean(env.finger_joint_positions))
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
        description="Evaluate IK learning model.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=5600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "evaluate_ik_learning.log"
    sys.stdout = Logger(log_file)

    input_dim = 9   # q_current(4) + pos_error(3) + dist(1) + phase(1)
    output_dim = 4  # dq (4)

    model = load_model(model_path, input_dim, output_dim, args.device)
    print(f"Loaded model from {model_path}")

    env = FrankaEnv(render_mode="human")

    results = []
    success_count = 0

    print(f"\nRunning {args.episodes} evaluation episodes...")
    for ep in range(args.episodes):
        ep_start = time.perf_counter()
        result = evaluate_one_episode(env, model, args.device, args.max_steps)
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

    # 保存结果
    results_data = {
        "episodes": results,
        "summary": {
            "success_rate": success_count / args.episodes,
            "success_count": success_count,
            "total_episodes": args.episodes,
            "avg_steps": np.mean([r["steps"] for r in results]),
            "avg_distance": np.mean([r["final_distance"] for r in results]),
        }
    }
    with open(output_dir / "evaluation_results_ik_learning.json", 'w') as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation Results (IK Learning)")
    print("=" * 60)
    print(f"  Success rate:     {success_count}/{args.episodes} ({success_count/args.episodes:.2%})")
    print(f"  Avg steps:        {np.mean([r['steps'] for r in results]):.1f}")
    print(f"  Avg final dist:   {np.mean([r['final_distance'] for r in results]):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
