from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from env import PROJECT_ROOT_DIR


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

    # 定义MLP模型
    obs_dim = train_obs.shape[1]
    act_dim = train_act.shape[1]
    model = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, act_dim),
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
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

    # 加载数据
    train_data, val_data = load_dataset_flat(data_dir)

    # 训练
    model = train_mlp(
        train_data,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=output_dir,
    )

    print(f"\nTraining complete. Model saved to {output_dir / 'model_best.pt'}")


if __name__ == "__main__":
    main()
