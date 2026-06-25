"""
可视化脚本 - 生成实验数据的图表，用于报告撰写。

Usage::
    python visualize.py --data_dir data/20260626_004329
    python visualize.py --data_dir data/20260626_004329 --output_dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_collection_results(data_dir: Path, output_dir: Path):
    """绘制数据采集结果图表"""
    # 加载数据
    experiment_file = data_dir / "experiment_data.json"
    statistics_file = data_dir / "episode_statistics.json"

    if not experiment_file.exists():
        print(f"警告: 未找到 {experiment_file}")
        return

    with open(experiment_file, 'r', encoding='utf-8') as f:
        experiment_data = json.load(f)

    if statistics_file.exists():
        with open(statistics_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    else:
        stats = None

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('数据采集结果分析', fontsize=16)

    # 1. 成功率随episode变化
    ax1 = axes[0, 0]
    episodes = experiment_data["episodes"]
    successes = [ep["success"] for ep in episodes]
    cumulative_success = []
    success_count = 0
    for i, success in enumerate(successes):
        success_count += int(success)
        cumulative_success.append(success_count / (i + 1))

    ax1.plot(range(1, len(cumulative_success) + 1), cumulative_success, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('累积成功率')
    ax1.set_title('累积成功率随Episode变化')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 2. 每个episode的步数
    ax2 = axes[0, 1]
    steps = [ep["steps"] for ep in episodes]
    ax2.bar(range(1, len(steps) + 1), steps, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('步数')
    ax2.set_title('每个Episode的步数')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 每个episode的耗时
    ax3 = axes[1, 0]
    durations = [ep["duration_seconds"] for ep in episodes]
    ax3.plot(range(1, len(durations) + 1), durations, 'g-', linewidth=1.5, marker='o', markersize=3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('耗时 (秒)')
    ax3.set_title('每个Episode的耗时')
    ax3.grid(True, alpha=0.3)

    # 4. 汇总统计
    ax4 = axes[1, 1]
    summary = experiment_data["summary"]
    bar_labels = ['成功', '失败', '总步数/100', '平均步数']
    bar_values = [
        summary["successful_episodes"],
        summary["failed_episodes"],
        summary["total_steps"] / 100,
        summary["avg_steps_per_episode"]
    ]
    colors = ['green', 'red', 'steelblue', 'orange']
    bars = ax4.bar(bar_labels, bar_values, color=colors, alpha=0.7)
    ax4.set_ylabel('数值')
    ax4.set_title('汇总统计')
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, bar_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "collection_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"数据采集图表已保存到: {output_path}")
    plt.close()


def plot_training_results(checkpoint_dir: Path, output_dir: Path):
    """绘制训练结果图表"""
    history_file = checkpoint_dir / "training_history.json"

    if not history_file.exists():
        print(f"警告: 未找到 {history_file}")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    history = data["history"]
    config = data["config"]

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('训练结果分析', fontsize=16)

    # 1. Loss曲线
    ax1 = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], 'b-', linewidth=2, label='训练Loss')
    ax1.plot(epochs, history["val_loss"], 'r-', linewidth=2, label='验证Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('训练和验证Loss曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标注最佳epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss = min(history["val_loss"])
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'最佳Epoch: {best_epoch}')
    ax1.annotate(f'最佳Loss: {best_loss:.6f}',
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + 5, best_loss + 0.001),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10)

    # 2. 学习率变化
    ax2 = axes[1]
    ax2.plot(epochs, history["learning_rate"], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('学习率')
    ax2.set_title('学习率调度')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "training_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"训练图表已保存到: {output_path}")
    plt.close()


def plot_evaluation_results(model_dir: Path, output_dir: Path):
    """绘制评估结果图表"""
    results_file = model_dir / "evaluation_results.json"

    if not results_file.exists():
        print(f"警告: 未找到 {results_file}")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    episodes = data["episodes"]
    summary = data["summary"]

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('评估结果分析', fontsize=16)

    # 1. 成功/失败分布
    ax1 = axes[0]
    success_count = summary["successful_episodes"]
    fail_count = summary["total_episodes"] - success_count
    labels = ['成功', '失败']
    sizes = [success_count, fail_count]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    if success_count > 0 or fail_count > 0:
        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        ax1.set_title(f'成功率: {summary["success_rate"]:.2%}')
    else:
        ax1.text(0.5, 0.5, '无评估数据', ha='center', va='center', fontsize=14)
        ax1.set_title('成功率')

    # 2. 每个episode的步数和距离
    ax2 = axes[1]
    steps = [ep["steps"] for ep in episodes]
    distances = [ep["final_distance"] for ep in episodes]

    x = range(1, len(steps) + 1)
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(x, steps, 'b-', linewidth=2, marker='o', markersize=5, label='步数')
    line2 = ax2_twin.plot(x, distances, 'r-', linewidth=2, marker='s', markersize=5, label='最终距离')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('步数', color='b')
    ax2_twin.set_ylabel('最终距离 (m)', color='r')
    ax2.set_title('每个Episode的步数和最终距离')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "evaluation_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"评估图表已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization charts for experiment results.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing experiment data (collect output).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing training checkpoints (default: data_dir/checkpoints).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: data_dir/figures).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else data_dir / "checkpoints"

    print("生成可视化图表...")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print()

    # 生成各类图表
    plot_collection_results(data_dir, output_dir)
    plot_training_results(checkpoint_dir, output_dir)
    plot_evaluation_results(checkpoint_dir, output_dir)

    print()
    print("=" * 60)
    print("所有图表已生成完毕！")
    print("=" * 60)
    print(f"图表保存在: {output_dir}")
    print("可以将这些图表插入到实验报告中。")


if __name__ == "__main__":
    main()
