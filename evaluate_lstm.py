"""
LSTM行为克隆评估脚本 - 使用时序模型进行评估。

Usage::
    python evaluate_lstm.py --model_path data/20260626_005525/checkpoints_lstm/model_best.pt --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import deque

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


def infer_stage(obs: np.ndarray) -> float:
    """根据观测推断当前阶段。"""
    finger_open = obs[17] if len(obs) > 17 else 0.0
    dist_ee_block = obs[18] if len(obs) > 18 else 0.5
    is_closed = obs[19] if len(obs) > 19 else 0.0

    if dist_ee_block > 0.3:
        if finger_open > 0.02:
            return 0.0
        else:
            return 0.6
    elif dist_ee_block > 0.15:
        if finger_open > 0.02:
            return 0.2
        else:
            return 0.6
    else:
        if is_closed > 0.5:
            return 0.4
        else:
            if finger_open > 0.02:
                return 0.8
            else:
                return 0.6


def add_stage_to_obs(obs: np.ndarray) -> np.ndarray:
    """在观测中添加阶段信息。"""
    stage = infer_stage(obs)
    return np.append(obs, [stage]).astype(np.float32)


class LSTMModel(torch.nn.Module):
    """LSTM行为克隆模型（与训练时一致）。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = torch.nn.LayerNorm(obs_dim)

        self.lstm = torch.nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        action = self.output_layers(last_output)
        return action


def load_model(model_path: Path, obs_dim: int, act_dim: int, device: str) -> LSTMModel:
    """加载训练好的LSTM模型。"""
    model = LSTMModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_one_episode(
    env: FrankaEnv,
    model: LSTMModel,
    device: str,
    seq_len: int = 32,
    max_steps: int = 5600,
    render: bool = True,
) -> Dict[str, float]:
    """运行一个评估episode。"""
    env.reset()

    success = False
    steps = 0

    # 使用双端队列存储历史观测
    obs_history = deque(maxlen=seq_len)

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

        # 添加阶段信息
        obs_with_stage = add_stage_to_obs(obs)

        # 添加到历史
        obs_history.append(obs_with_stage)

        # 如果历史不够长，使用零填充
        if len(obs_history) < seq_len:
            # 用第一个观测填充
            padding = [obs_history[0]] * (seq_len - len(obs_history))
            window = np.array(padding + list(obs_history))
        else:
            window = np.array(list(obs_history))

        # 模型预测
        with torch.no_grad():
            obs_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            pred_act = model(obs_tensor).squeeze(0).cpu().numpy()

        # 应用动作
        arm_action = pred_act[:4]
        grip_action = np.clip(pred_act[4], 0.0, 1.0)

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
        description="Evaluate LSTM behavior cloning model.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=5600)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
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
    log_file = output_dir / "evaluate_lstm.log"
    sys.stdout = Logger(log_file)

    # 观测和动作维度
    obs_dim = 21  # 原始20 + 阶段1
    act_dim = 5

    # 加载模型
    model = load_model(model_path, obs_dim, act_dim, args.device)
    print(f"Loaded model from {model_path}")
    print(f"Sequence length: {args.seq_len}")

    # 初始化环境
    env = FrankaEnv(render_mode="human")

    # 评估
    results = []
    success_count = 0

    print(f"\nRunning {args.episodes} evaluation episodes...")
    for ep in range(args.episodes):
        ep_start = time.perf_counter()
        result = evaluate_one_episode(
            env, model, args.device, args.seq_len, args.max_steps, render=True
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

    # 保存结果
    results_data = {
        "config": {
            "model_path": str(model_path),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seq_len": args.seq_len,
        },
        "episodes": results,
        "summary": {
            "success_rate": success_count / args.episodes,
            "success_count": success_count,
            "total_episodes": args.episodes,
            "avg_steps": np.mean([r["steps"] for r in results]),
            "avg_distance": np.mean([r["final_distance"] for r in results]),
        }
    }

    results_path = output_dir / "evaluation_results_lstm.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    # 汇总结果
    print("\n" + "=" * 60)
    print("Evaluation Results (LSTM)")
    print("=" * 60)
    print(f"  Success rate:     {success_count}/{args.episodes} ({success_count/args.episodes:.2%})")
    print(f"  Avg steps:        {np.mean([r['steps'] for r in results]):.1f}")
    print(f"  Avg final dist:   {np.mean([r['final_distance'] for r in results]):.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
