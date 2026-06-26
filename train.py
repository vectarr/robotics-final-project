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


class TrainingRecorder:
    """训练过程记录器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_times": [],
        }
        self.config = {}
        self.start_time = None

    def save_config(self, args, obs_dim, act_dim):
        """保存训练配置"""
        self.config = {
            "data_dir": str(args.data_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "model_architecture": "MLP(256, 128)",
            "start_time": datetime.now().isoformat(),
        }

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float,
                     lr: float, epoch_time: float):
        """记录单个epoch的数据"""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["learning_rate"].append(lr)
        self.history["epoch_times"].append(epoch_time)

    def save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "training_history.json"
        data = {
            "config": self.config,
            "history": self.history,
            "summary": {
                "total_epochs": len(self.history["train_loss"]),
                "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
                "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
                "best_val_loss": min(self.history["val_loss"]) if self.history["val_loss"] else None,
                "total_time_seconds": sum(self.history["epoch_times"]),
                "end_time": datetime.now().isoformat(),
            }
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"训练历史已保存到: {history_path}")


# ---------------------------------------------------------------------------
# Data loading — flat (MLP)
# ---------------------------------------------------------------------------
def load_dataset_flat(
    data_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Load all episodes, concatenate into flat (N, D) tensors (for MLP)."""
    files = sorted(data_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")

    print(f"Loading {len(files)} episodes from {data_dir} …")

    all_obs, all_act = [], []
    for f in files:
        d = np.load(f)
        all_obs.append(d["observations"])
        all_act.append(d["actions"])

    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    act = np.concatenate(all_act, axis=0).astype(np.float32)

    print(f"  Total samples: {obs.shape[0]:,}  obs={obs.shape}  act={act.shape}")

    rng = np.random.RandomState(seed)
    n_total = obs.shape[0]
    n_val = max(1, int(n_total * val_split))
    idx = rng.permutation(n_total)

    train_obs = torch.from_numpy(obs[idx[n_val:]])
    train_act = torch.from_numpy(act[idx[n_val:]])
    val_obs = torch.from_numpy(obs[idx[:n_val]])
    val_act = torch.from_numpy(act[idx[:n_val]])

    print(f"  Train: {len(train_obs):,}  Val: {len(val_obs):,}")
    return (train_obs, train_act), (val_obs, val_act)


# ---------------------------------------------------------------------------
# Training — MLP
# ---------------------------------------------------------------------------
def train_mlp(
    train_data,
    val_data,
    *,
    epochs,
    batch_size,
    lr,
    weight_decay,
    device,
    checkpoint_dir,
    recorder=None,
):
    from torch.utils.data import DataLoader, TensorDataset

    train_obs, train_act = train_data
    val_obs, val_act = val_data

    train_loader = DataLoader(
        TensorDataset(train_obs, train_act),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(val_obs, val_act),
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # 定义MLP模型（更大的网络）
    obs_dim = train_obs.shape[1]
    act_dim = train_act.shape[1]
    model = nn.Sequential(
        nn.Linear(obs_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, act_dim),
        # 不使用Sigmoid，因为关节动作需要负值和大于1的值
        # 夹爪动作在推理时单独处理
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        train_loss_sum, batches = 0.0, 0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)
            loss = loss_fn(pred, act_b)
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
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                val_loss_sum += loss_fn(model(obs_b), act_b).item()
                v_batches += 1
        val_loss = val_loss_sum / max(v_batches, 1)

        epoch_time = time.perf_counter() - epoch_start

        # Record epoch data
        if recorder:
            recorder.record_epoch(epoch, train_loss, val_loss,
                                 scheduler.get_last_lr()[0], epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_dir:
                torch.save(model.state_dict(), checkpoint_dir / "model_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"  epoch {epoch:4d}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s",
                flush=True,
            )

    if checkpoint_dir:
        model.load_state_dict(
            torch.load(
                checkpoint_dir / "model_best.pt", map_location=device, weights_only=True
            )
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train behavior cloning model on collected data.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing episode_*.npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: data_dir/checkpoints).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (default: cuda if available).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = output_dir / "train.log"
    sys.stdout = Logger(log_file)

    # Setup recorder
    recorder = TrainingRecorder(output_dir)

    # 加载数据
    train_data, val_data = load_dataset_flat(data_dir)
    obs_dim = train_data[0].shape[1]
    act_dim = train_data[1].shape[1]
    recorder.save_config(args, obs_dim, act_dim)

    # 训练
    print(f"\nStarting training...")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print()

    model = train_mlp(
        train_data,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=output_dir,
        recorder=recorder,
    )

    # Save training history
    recorder.save_history()

    print(f"\nTraining complete. Model saved to {output_dir / 'model_best.pt'}")
    print(f"Training history saved to {output_dir / 'training_history.json'}")
    print(f"Log file saved to {log_file}")


if __name__ == "__main__":
    main()
