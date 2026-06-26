"""
IK学习训练脚本 - 直接学习IK控制器的Jacobian计算。

核心思路：
- 输入：当前状态 + 目标位置
- 输出：关节增量 dq（而非绝对关节位置）
- 这样模型只需要学会"如何移动"

Usage::
    python train_ik_learning.py --data_dir data/20260626_005525 --epochs 200
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


def compute_ik_targets(data_dir: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    从训练数据中计算IK目标和关节增量。

    返回：[(观测, 关节增量), ...] 的列表
    """
    files = sorted(data_dir.glob("episode_*.npz"))
    all_samples = []

    for f in files[:200]:  # 限制episode数量
        data = np.load(f)
        obs = data["observations"]  # (T, 20)
        act = data["actions"]       # (T, 5)

        q_current = obs[:, :4]   # 当前关节位置
        q_target = act[:, :4]    # IK计算的目标位置
        dq = q_target - q_current  # 关节增量

        # 计算误差向量
        ee_pos = obs[:, 8:11]    # 末端位置
        blk_pos = obs[:, 11:14]  # 方块位置
        tgt_pos = obs[:, 14:17]  # 目标位置
        finger_open = obs[:, 17]  # 夹爪开度
        dist_ee_block = obs[:, 18]  # 到方块距离

        # 根据阶段确定目标位置
        # 规则：距离<0.15且夹爪闭合时，目标是tgt_pos；否则是blk_pos
        target_pos = np.zeros_like(ee_pos)
        for i in range(len(obs)):
            if dist_ee_block[i] < 0.15 and finger_open[i] < 0.02:
                # LIFT + MOVE + PLACE阶段：目标是目标位置
                target_pos[i] = tgt_pos[i]
            else:
                # APPROACH + DESCEND阶段：目标是方块位置
                target_pos[i] = blk_pos[i]

        # 计算位置误差
        pos_error = target_pos - ee_pos  # (T, 3)

        # 构建输入特征
        # [q_current, pos_error, dist_to_target, phase]
        dist_to_target = np.linalg.norm(pos_error, axis=1, keepdims=True)

        # 添加阶段信息
        phase = np.zeros((len(obs), 1))
        for i in range(len(obs)):
            if dist_ee_block[i] > 0.3:
                phase[i] = 0.0  # APPROACH
            elif dist_ee_block[i] > 0.15:
                phase[i] = 0.2  # DESCEND
            elif finger_open[i] < 0.02:
                phase[i] = 0.5  # GRASP + LIFT + MOVE
            else:
                phase[i] = 0.8  # PLACE

        features = np.concatenate([
            q_current,      # 4
            pos_error,      # 3
            dist_to_target, # 1
            phase,          # 1
        ], axis=1)  # (T, 9)

        for i in range(len(features)):
            all_samples.append((features[i], dq[i]))

    return all_samples


class IKDataset(Dataset):
    """IK学习数据集。"""

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]]):
        self.features = np.array([s[0] for s in samples], dtype=np.float32)
        self.targets = np.array([s[1] for s in samples], dtype=np.float32)

        print(f"  样本数量: {len(self)}")
        print(f"  输入维度: {self.features.shape[1]}")
        print(f"  输出维度: {self.targets.shape[1]}")
        print(f"  目标范围: [{self.targets.min():.4f}, {self.targets.max():.4f}]")
        print(f"  目标均值: {self.targets.mean(axis=0)}")
        print(f"  目标标准差: {self.targets.std(axis=0)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class IKModel(nn.Module):
    """IK学习模型 - 预测关节增量。"""

    def __init__(self, input_dim: int = 9, output_dim: int = 4):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh(),  # 输出在[-1, 1]范围内
        )

        # 输出缩放（关节增量通常在[-0.02, 0.02]范围内）
        self.output_scale = 0.02

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) * self.output_scale


def train_ik(
    train_dataset: IKDataset,
    val_dataset: IKDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    checkpoint_dir: Path,
):
    """训练IK学习模型。"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
    )

    input_dim = train_dataset.features.shape[1]
    output_dim = train_dataset.targets.shape[1]

    model = IKModel(input_dim=input_dim, output_dim=output_dim)
    model.to(device)

    print(f"\n模型结构:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  设备: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    t_start = time.perf_counter()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, batches = 0.0, 0
        for feat_b, target_b in train_loader:
            feat_b, target_b = feat_b.to(device), target_b.to(device)
            pred = model(feat_b)
            loss = loss_fn(pred, target_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            batches += 1
        scheduler.step()
        train_loss = train_loss_sum / max(batches, 1)

        model.eval()
        val_loss_sum, v_batches = 0.0, 0
        with torch.no_grad():
            for feat_b, target_b in val_loader:
                feat_b, target_b = feat_b.to(device), target_b.to(device)
                val_loss_sum += loss_fn(model(feat_b), target_b).item()
                v_batches += 1
        val_loss = val_loss_sum / max(v_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "model_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.8f}  val_loss={val_loss:.8f}  "
                f"{elapsed:.1f}s",
                flush=True,
            )

    model.load_state_dict(
        torch.load(checkpoint_dir / "model_best.pt", map_location=device, weights_only=True)
    )

    # 保存历史
    with open(checkpoint_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IK learning model.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "checkpoints_ik"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "train.log"
    sys.stdout = Logger(log_file)

    print("=" * 60)
    print("IK学习训练 - 直接学习Jacobian计算")
    print("=" * 60)

    # 准备数据
    print("\n准备数据...")
    samples = compute_ik_targets(data_dir)

    # 划分训练集和验证集
    rng = np.random.RandomState(42)
    rng.shuffle(samples)
    n_val = len(samples) // 10
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"\n创建训练集...")
    train_dataset = IKDataset(train_samples)

    print(f"\n创建验证集...")
    val_dataset = IKDataset(val_samples)

    print(f"\n训练配置:")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")

    # 训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    model = train_ik(
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=output_dir,
    )

    print("\n训练完成！")
    print(f"模型保存在: {output_dir / 'model_best.pt'}")


if __name__ == "__main__":
    main()
