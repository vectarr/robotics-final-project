"""
LSTM行为克隆训练脚本 - 学习时序决策能力。

Usage::
    python train_lstm.py --data_dir data/20260626_005525 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import PROJECT_ROOT_DIR


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
    """根据观测推断当前阶段（0-1之间的值）。

    阶段编码：
    - 0.0: APPROACH（接近阶段，距离远，夹爪打开）
    - 0.2: DESCEND（下降阶段，距离中等，夹爪打开）
    - 0.4: GRASP（抓取阶段，距离近，夹爪关闭）
    - 0.6: LIFT+MOVE（移动阶段，距离近，夹爪关闭）
    - 0.8: PLACE（放置阶段，距离近，夹爪打开）
    - 1.0: RETREAT（撤离阶段，距离远，夹爪打开）
    """
    finger_open = obs[17] if len(obs) > 17 else 0.0
    dist_ee_block = obs[18] if len(obs) > 18 else 0.5
    is_closed = obs[19] if len(obs) > 19 else 0.0

    # 基于距离和夹爪状态推断阶段
    if dist_ee_block > 0.3:
        if finger_open > 0.02:
            return 0.0  # APPROACH
        else:
            return 0.6  # LIFT+MOVE（可能是在移动中）
    elif dist_ee_block > 0.15:
        if finger_open > 0.02:
            return 0.2  # DESCEND
        else:
            return 0.6  # LIFT+MOVE
    else:  # dist_ee_block <= 0.15
        if is_closed > 0.5:
            return 0.4  # GRASP
        else:
            if finger_open > 0.02:
                return 0.8  # PLACE
            else:
                return 0.6  # LIFT+MOVE


def add_stage_to_obs(obs: np.ndarray) -> np.ndarray:
    """在观测中添加阶段信息。"""
    stage = infer_stage(obs)
    # 原始obs: 20维
    # 添加阶段: 21维
    return np.append(obs, [stage]).astype(np.float32)


class SequenceDataset(Dataset):
    """时序数据集 - 使用滑动窗口创建序列。"""

    def __init__(self, sequences: List[np.ndarray], actions: List[np.ndarray],
                 seq_len: int = 32):
        """
        Args:
            sequences: 观测序列列表，每个元素形状为 (T, obs_dim)
            actions: 动作序列列表，每个元素形状为 (T, act_dim)
            seq_len: 序列长度
        """
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        for seq, act_seq in zip(sequences, actions):
            # 为每个episode创建多个训练样本
            for t in range(seq_len, len(seq)):
                # 输入：过去seq_len步的观测
                window = seq[t-seq_len:t]
                # 标签：当前步的动作
                label = act_seq[t]
                self.data.append(window)
                self.labels.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        print(f"  创建了 {len(self.data)} 个训练样本")
        print(f"  输入形状: {self.data.shape}  (batch, seq_len, obs_dim)")
        print(f"  标签形状: {self.labels.shape}  (batch, act_dim)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LSTMModel(nn.Module):
    """LSTM行为克隆模型。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入层归一化
        self.input_norm = nn.LayerNorm(obs_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch, seq_len, obs_dim)

        Returns:
            动作预测，形状为 (batch, act_dim)
        """
        # 归一化输入
        x = self.input_norm(x)

        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 输出层
        action = self.output_layers(last_output)

        return action


def load_dataset_lstm(
    data_dir: Path,
    seq_len: int = 32,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[SequenceDataset, SequenceDataset]:
    """加载数据并创建时序数据集。"""
    files = sorted(data_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")

    print(f"Loading {len(files)} episodes from {data_dir} …")

    all_sequences = []
    all_actions = []

    for f in files:
        d = np.load(f)
        obs = d["observations"]
        act = d["actions"]

        # 在观测中添加阶段信息
        obs_with_stage = np.array([add_stage_to_obs(o) for o in obs])

        all_sequences.append(obs_with_stage)
        all_actions.append(act)

    print(f"  观测维度: {all_sequences[0].shape[1]} (原始20 + 阶段1 = 21)")
    print(f"  动作维度: {all_actions[0].shape[1]}")

    # 划分训练集和验证集
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_split))

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    print(f"\n创建训练集...")
    train_dataset = SequenceDataset(
        [all_sequences[i] for i in train_indices],
        [all_actions[i] for i in train_indices],
        seq_len=seq_len,
    )

    print(f"\n创建验证集...")
    val_dataset = SequenceDataset(
        [all_sequences[i] for i in val_indices],
        [all_actions[i] for i in val_indices],
        seq_len=seq_len,
    )

    return train_dataset, val_dataset


def train_lstm(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    checkpoint_dir: Path,
    obs_dim: int,
    act_dim: int,
    seq_len: int,
):
    """训练LSTM模型。"""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # 创建模型
    model = LSTMModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
    )
    model.to(device)

    print(f"\n模型结构:")
    print(f"  输入维度: {obs_dim}")
    print(f"  输出维度: {act_dim}")
    print(f"  隐藏层大小: 256")
    print(f"  LSTM层数: 2")
    print(f"  序列长度: {seq_len}")
    print(f"  设备: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    t_start = time.perf_counter()
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, batches = 0.0, 0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)
            loss = loss_fn(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            batches += 1
        scheduler.step()
        train_loss = train_loss_sum / max(batches, 1)

        model.eval()
        val_loss_sum, v_batches = 0.0, 0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                val_loss_sum += loss_fn(model(obs_b), act_b).item()
                v_batches += 1
        val_loss = val_loss_sum / max(v_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "model_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s",
                flush=True,
            )

    # 加载最佳模型
    model.load_state_dict(
        torch.load(checkpoint_dir / "model_best.pt", map_location=device, weights_only=True)
    )

    # 保存训练历史
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LSTM behavior cloning model.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=32,
                       help="LSTM序列长度（历史步数）")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "checkpoints_lstm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_file = output_dir / "train.log"
    sys.stdout = Logger(log_file)

    print("=" * 60)
    print("LSTM行为克隆训练")
    print("=" * 60)

    # 加载数据
    train_dataset, val_dataset = load_dataset_lstm(
        data_dir, seq_len=args.seq_len, val_split=0.1
    )

    # 获取维度
    obs_dim = train_dataset.data.shape[2]  # (batch, seq_len, obs_dim)
    act_dim = train_dataset.labels.shape[1]

    print(f"\n训练配置:")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {act_dim}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  设备: {args.device}")

    # 训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    model = train_lstm(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=output_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        seq_len=args.seq_len,
    )

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {output_dir / 'model_best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
