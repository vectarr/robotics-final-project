#!/bin/bash
# A100服务器环境配置脚本（使用conda）
# 服务器配置：4x NVIDIA A100-SXM4-40GB, CUDA 12.9

set -e

echo "=========================================="
echo "  A100服务器环境配置"
echo "=========================================="

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Miniconda或Anaconda"
    exit 1
fi

# 检查NVIDIA驱动
echo ""
echo "[1/6] 检查NVIDIA驱动..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 创建conda环境
echo ""
echo "[2/6] 创建conda环境..."
ENV_NAME="robotics_Imitation Learning"
conda create -n $ENV_NAME python=3.10 -y
echo "环境 $ENV_NAME 创建完成"

# 激活环境
echo ""
echo "[3/6] 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装PyTorch（CUDA 12.9兼容版本）
echo ""
echo "[4/6] 安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
echo ""
echo "[5/6] 安装其他依赖..."
pip install numpy mujoco matplotlib

# 验证安装
echo ""
echo "[6/6] 验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import mujoco; print(f'MuJoCo版本: {mujoco.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib版本: {matplotlib.__version__}')"

echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "  1. 激活环境: conda activate $ENV_NAME"
echo "  2. 进入项目目录: cd /data/yuwj/linhuajie/robotics-final-project"
echo "  3. 开始训练: python train_lstm.py --data_dir data/20260626_005525 --epochs 200"
echo ""
echo "环境名称: $ENV_NAME"
echo "Python版本: 3.10"
echo "PyTorch: CUDA 12.4兼容版本"
