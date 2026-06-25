"""
调试脚本 - 分析模型行为和问题。

Usage::
    python debug_model.py --model_path data/20260626_005525/checkpoints/model_best.pt --data_dir data/20260626_005525
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def analyze_model_predictions(model_path: Path, data_dir: Path, device: str = "cpu"):
    """分析模型在训练数据上的预测"""

    # 加载模型
    from train import load_dataset_flat
    train_data, val_data = load_dataset_flat(data_dir)
    train_obs, train_act = train_data

    obs_dim = train_obs.shape[1]
    act_dim = train_act.shape[1]

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

    # 随机采样一些训练样本
    np.random.seed(42)
    indices = np.random.choice(len(train_obs), 10, replace=False)

    print("模型在训练数据上的预测对比:")
    print("=" * 70)

    for i, idx in enumerate(indices):
        obs = train_obs[idx:idx+1].to(device)
        target_act = train_act[idx].cpu().numpy()

        with torch.no_grad():
            pred_act = model(obs).squeeze(0).cpu().numpy()

        error = np.abs(pred_act - target_act).mean()

        print(f"\n样本 {i+1}:")
        print(f"  真实动作: {target_act}")
        print(f"  预测动作: {pred_act}")
        print(f"  平均误差: {error:.6f}")

    # 检查模型输出范围
    print("\n" + "=" * 70)
    print("模型输出范围分析:")
    print("=" * 70)

    with torch.no_grad():
        # 使用训练数据的最小值和最大值
        min_obs = train_obs.min(dim=0).values.unsqueeze(0).to(device)
        max_obs = train_obs.max(dim=0).values.unsqueeze(0).to(device)

        pred_min = model(min_obs).squeeze(0).cpu().numpy()
        pred_max = model(max_obs).squeeze(0).cpu().numpy()

        print(f"  输入范围 (min): {min_obs.squeeze().cpu().numpy()}")
        print(f"  输入范围 (max): {max_obs.squeeze().cpu().numpy()}")
        print(f"  预测输出 (min输入): {pred_min}")
        print(f"  预测输出 (max输入): {pred_max}")

    # 检查训练数据的动作范围
    print("\n" + "=" * 70)
    print("训练数据动作范围:")
    print("=" * 70)
    print(f"  动作最小值: {train_act.min(dim=0).values.numpy()}")
    print(f"  动作最大值: {train_act.max(dim=0).values.numpy()}")
    print(f"  动作均值: {train_act.mean(dim=0).numpy()}")
    print(f"  动作标准差: {train_act.std(dim=0).numpy()}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug model behavior.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference.",
    )
    args = parser.parse_args()

    analyze_model_predictions(Path(args.model_path), Path(args.data_dir), args.device)


if __name__ == "__main__":
    main()
