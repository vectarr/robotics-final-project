#!/bin/bash

# 云端环境配置脚本
# 使用方法: bash setup.sh

set -e

echo "=========================================="
echo "  机器人导论大作业 - 环境配置"
echo "=========================================="

# 检查Python版本
echo ""
echo "[1/4] 检查Python版本..."
python3 --version || { echo "错误: 未找到Python3"; exit 1; }

# 检查CUDA
echo ""
echo "[2/4] 检查CUDA..."
if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
else
    echo "警告: PyTorch未安装或CUDA不可用"
fi

# 安装依赖
echo ""
echo "[3/4] 安装依赖..."
pip install --upgrade pip

# 检测CUDA版本并安装对应的PyTorch
echo "检测CUDA版本..."
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2-4)
echo "检测到CUDA版本: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "安装PyTorch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" || "$CUDA_VERSION" == "12.2" || "$CUDA_VERSION" == "12.4" ]]; then
    echo "安装PyTorch for CUDA 12.x..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "使用默认PyTorch安装..."
    pip install torch torchvision torchaudio
fi

pip install numpy mujoco matplotlib

# 检查MuJoCo
echo ""
echo "[4/4] 检查MuJoCo..."
python3 -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"

echo ""
echo "=========================================="
echo "  配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 数据采集: python collect.py --episodes 100 --rand"
echo "  2. 训练模型: python train.py --data_dir data/<数据目录>"
echo "  3. 评估模型: python evaluate.py --model_path checkpoints/model_best.pt"
echo ""
echo "详细说明请查看 README.md"
